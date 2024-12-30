'''Collect the attacking model and surrogate model'''
import json
import torch
import wandb
import pickle
import numpy as np
from torch import nn
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from research.utils.generation import top_filtering
from research.utils.optim import PairwisePivotLoss, SequenceCrossEntropyLoss, prepare_optimizer
from research.utils.eval import calculate_metrics
from research.utils.toolbox import create_save_path, create_save_dir_path
from research.data.data_processing import get_document_embs
from sentence_transformers import SentenceTransformer


class LinearProjection(nn.Module):
    '''Linear projection layer to project the embedding to the same dim as LM'''

    def __init__(self, in_num, out_num):
        super().__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, embs):
        '''Forward the embedding to the projection layer'''
        return torch.clamp(self.fc1(embs), min=-1e9, max=1e9)


class MappingNetwork(nn.Module):
    '''Define the mapping networks to map embeddings into a common space'''

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, (input_dim + output_dim) // 2),
            nn.ReLU(),
            nn.Linear((input_dim + output_dim) // 2, output_dim)
        )

    def forward(self, embs):
        '''Forward the embedding to the mapping network'''
        return self.net(embs.float())


class Discriminator(nn.Module):
    '''Define a discriminator network for the common space'''

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, embs):
        '''Forward the embedding to the discriminator network'''
        return self.net(embs)


class LLMAttackModel():
    '''LM based attack model'''

    def __init__(self, config, emb_dim, emb2_dim, device):
        self.config = config
        self.emb_dim = emb_dim
        self.device = device
        # Decoder model
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_dir']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_dir'])
        self.projection = LinearProjection(
            in_num=self.emb_dim, out_num=self.model.config.hidden_size).to(self.device)
        self.criterion = SequenceCrossEntropyLoss()
        self.optimizer = prepare_optimizer(self.model)
        self.optimizer.add_param_group(
            {'params': self.projection.parameters()}
        )

        # Pivot training
        self.mse_loss = torch.nn.MSELoss()
        self.pairwise_pivot_loss = PairwisePivotLoss()

        # Adversarial training
        self.bce_loss = torch.nn.BCELoss()
        self.mapping = MappingNetwork(emb2_dim, emb_dim).to(self.device)
        self.discriminator = Discriminator(emb_dim).to(self.device)
        self.optimizer.add_param_group(
            {'params': self.mapping.parameters()}
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.001)

    def fit(self, adv_train_loader, val_loader) -> list[float]:
        '''Training the attack model'''
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=100,
                                                    num_training_steps=len(
                                                        adv_train_loader) * self.config['num_epochs'])
        step = 0
        result_score = []
        for epoch in trange(self.config['num_epochs']):
            # Train the discriminator
            if not self.config['geia']:
                self.mapping.requires_grad_(False)
                self.discriminator.requires_grad_(True)
                for _, (batch_text, embs, s_embs, domains) in enumerate(adv_train_loader):
                    # Separate embeddings based on domain
                    embs = embs.to(self.device)
                    s_embs = s_embs.to(self.device)
                    emb1 = embs[domains == 0]
                    emb2 = self.mapping(s_embs[domains == 1])
                    dis_loss = self.discriminator_on_epoch(emb1, emb2)

                    # Update discriminator
                    self.d_optimizer.zero_grad()
                    dis_loss.backward()
                    self.d_optimizer.step()

            # Train the decoder and mapping network
            self.mapping.requires_grad_(True)
            self.discriminator.requires_grad_(False)
            train_loss, train_perplexity = 0, 0
            for _, (batch_text, embs, s_embs, domains) in enumerate(adv_train_loader):
                # Separate embeddings based on domain
                text1 = [batch_text[i]
                         for i in range(len(domains)) if domains[i] == 0]
                text2 = [batch_text[i]
                         for i in range(len(domains)) if domains[i] == 1]
                # Private embeddings don't need to be mapped
                emb1 = embs[domains == 0].to(self.device)
                # Switch between domain adversarial training and normal training
                if not self.config['geia']:
                    # Surrogate embeddings need to be mapped to the same space
                    s_emb1 = self.mapping(s_embs[domains == 0].to(self.device))
                    emb2 = self.mapping(s_embs[domains == 1].to(self.device))

                    # Calculate loss
                    pivot_loss = self.pivot_on_batch(emb1, s_emb1)
                    map_loss = self.mapping_on_batch(emb2)
                    train_loss, train_perplexity = self.train_on_batch(
                        embeddings=self.projection(torch.cat([emb1, emb2])), text=text1 + text2)
                    loss = train_loss + \
                        self.config['mapping_lambda'] * map_loss + \
                        self.config['pivot_lambda'] * pivot_loss
                else:
                    train_loss, train_perplexity = self.train_on_batch(
                        embeddings=self.projection(emb1), text=text1)
                    loss = train_loss

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                scheduler.step()

            # Evaluate on validation set
            if (epoch + 1) % self.config['eval_per_epochs'] == 0:
                result = self.eval_on_batch(val_loader, step)
                print(
                    f"Epoch:[{epoch+1}/{self.config['num_epochs']}], Train_loss: {train_loss.item()}, Train_perplexity: {train_perplexity}")
                result['train/loss'] = train_loss.item()
                result['train/perplexity'] = train_perplexity
                print(result)
                if not self.config['testing']:
                    wandb.log(result)

                result_score.append(result)
                self.save_models(step)
                step += 1

        return result_score

    def pivot_on_batch(self, emb1, emb2):
        '''Calculate two kinds of pivot losses'''
        same_pair_loss = self.mse_loss(emb1, emb2)
        pairwise_loss = self.pairwise_pivot_loss(emb1, emb2)

        return same_pair_loss + pairwise_loss

    def discriminator_on_epoch(self, emb1, emb2):
        '''Train the discriminator on a batch of data'''
        predictions1 = self.discriminator(emb1)
        predictions2 = self.discriminator(emb2)

        # private examples have label 0
        labels1 = torch.zeros(predictions1.size())
        # surrogate examples have label 1
        labels2 = torch.ones(predictions2.size())

        predictions = torch.cat((predictions1, predictions2), dim=0)
        labels = torch.cat((labels1, labels2), dim=0).to(self.device)
        d_loss = self.bce_loss(predictions, labels)
        return d_loss

    def mapping_on_batch(self, emb2):
        '''Train the mapping network to deceive the discriminator'''
        predictions2 = self.discriminator(emb2)
        # mapped examples have label 0, opposite of the discriminator's expectation
        labels2 = torch.zeros(predictions2.size()).to(self.device)
        map_loss = self.bce_loss(predictions2, labels2)
        return map_loss

    def train_on_batch(self, embeddings, text):
        '''Train on a batch of data'''
        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_ids = self.tokenizer(text, return_tensors='pt',
                                   padding='max_length', truncation=True,
                                   max_length=40)['input_ids'].to(self.device)
        labels = input_ids.clone()
        # print(input_ids.size())
        # embed the input ids using GPT-2 embedding
        input_emb = self.model.transformer.wte(input_ids)
        # add extra dim to cat together
        embeddings = embeddings.to(self.device)
        embeddings = torch.unsqueeze(embeddings, 1)
        # [batch,max_length+1,emb_dim (1024)]
        inputs_embeds = torch.cat((embeddings, input_emb), dim=1)
        past = None

        # logits, past = model(inputs_embeds=inputs_embeds,past = past)
        logits, past = self.model(inputs_embeds=inputs_embeds,
                                  past_key_values=past, return_dict=False)
        logits = logits[:, :-1].contiguous()
        target = labels.contiguous()
        target_mask = torch.ones_like(target).float()
        loss = self.criterion(logits, target, target_mask,
                              label_smoothing=0.02, reduce="batch")

        record_loss = loss.item()
        perplexity = np.exp(record_loss)
        return loss, perplexity

    def predict(self, dataloader, step):
        '''Predict on a batch of data'''
        sent_dict = {}
        sent_dict['gt'] = []
        sent_dict['pred'] = []
        self.load_models(step)
        with torch.no_grad():
            for _, (batch_text, embeddings) in enumerate(dataloader):

                embeddings = embeddings.to(self.device)
                embeddings = self.projection(embeddings)
                sent_list, gt_list = self.inference_on_batch(
                    embeddings=embeddings, sentences=batch_text)
                sent_dict['pred'].extend(sent_list)
                sent_dict['gt'].extend(gt_list)

            self.save_prediction(sent_dict, step)

    def save_prediction(self, sent_dict, step):
        '''Save the prediction to file'''
        output_path = create_save_path(self.config, step)
        print(f"Saving result to: {output_path}")
        with open(output_path, 'w', encoding='UTF-8') as output_file:
            json.dump(sent_dict, output_file, indent=4)

    def eval_on_batch(self, dataloader, epoch):
        '''Evaluate on a batch of data'''
        sent_dict = {}
        sent_dict['gt'] = []
        sent_dict['pred'] = []
        perplexity_list = []
        with torch.no_grad():
            for _, (batch_text, embeddings) in enumerate(dataloader):
                embeddings = embeddings.to(self.device)
                embeddings = self.projection(embeddings)
                # Calculate the perplexity
                _, val_perplexity = self.train_on_batch(
                    embeddings=embeddings, text=batch_text)
                perplexity_list.append(val_perplexity)
                # Generate the sentence
                sent_list, gt_list = self.inference_on_batch(
                    embeddings=embeddings, sentences=batch_text)
                sent_dict['pred'].extend(sent_list)
                sent_dict['gt'].extend(gt_list)

        # Save the prediction to file
        self.save_prediction(sent_dict, epoch)
        # Evaluate the result
        metrics = calculate_metrics(sent_dict['pred'], sent_dict['gt'])
        metrics.update({'perplexity': np.mean(perplexity_list)})
        return metrics

    def inference_on_batch(self, embeddings, sentences):
        '''Evaluate on a batch of data'''
        decode_method = self.config['decode']
        embeddings = embeddings.to(self.device)
        sent_list = []
        gt_list = sentences
        for _, hidden in enumerate(embeddings):
            if decode_method == 'sampling':
                sentence = self.generate_sentence(
                    hidden_embedding=hidden)
            else:
                pass
            sent_list.append(sentence)

        return sent_list, gt_list

    def generate_sentence(self, hidden_embedding):
        '''Generate sentence using LLM'''
        temperature = 0.9
        top_k = -1
        top_p = 0.9
        sent = []
        prev_input = None
        past = None
        eos = self.tokenizer.encode("<|endoftext|>")
        hidden_embedding = torch.unsqueeze(hidden_embedding, 0)
        hidden_embedding = torch.unsqueeze(
            hidden_embedding, 0)  # [1,1,embed_dim]
        logits, past = self.model(inputs_embeds=hidden_embedding,
                                  past_key_values=past, return_dict=False)
        logits = logits[:, -1, :] / temperature
        logits = torch.clamp(logits, min=-1e9, max=1e9)
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)

        prev_input = torch.multinomial(probs, num_samples=1)
        prev_word = prev_input.item()
        sent.append(prev_word)

        for _ in range(50):
            # logits, past = model(prev_input, past=past)
            logits, past = self.model(
                prev_input, past_key_values=past, return_dict=False)
            logits = logits[:, -1, :] / temperature
            logits = torch.clamp(logits, min=-1e9, max=1e9)
            logits = top_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)

            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == eos[0]:
                break
            sent.append(prev_word)

        output = self.tokenizer.decode(sent)

        return output

    def save_models(self, step):
        '''Save the model to the path'''
        save_dir = create_save_dir_path(self.config, step)
        map_path = f"{save_dir}/" + \
            f"mapping_{self.config['dataset']}_{self.config['blackbox_encoder']}_{self.config['surrogate_encoder']}"
        proj_path = f"{save_dir}/" + \
            f"projection_{self.config['dataset']}_{self.config['blackbox_encoder']}_{self.config['surrogate_encoder']}"
        attacker_path = f"{save_dir}/" + \
            f"attacler_{self.config['dataset']}_{self.config['blackbox_encoder']}_{self.config['surrogate_encoder']}"

        print(f"Saving model to: {save_dir}")
        print(f"Saving mapping to: {map_path}")
        print(f"Saving projection to: {proj_path}")
        print(f"Saving attacker to: {attacker_path}")
        torch.save(self.mapping.state_dict(), map_path)
        torch.save(self.projection.state_dict(), proj_path)
        self.model.save_pretrained(attacker_path)

    def load_models(self, step):
        '''Load the model from the path'''
        save_dir = create_save_dir_path(self.config, step)
        map_path = f"{save_dir}/" + \
            f"mapping_{self.config['dataset']}_{self.config['blackbox_encoder']}_{self.config['surrogate_encoder']}"
        proj_path = f"{save_dir}/" + \
            f"projection_{self.config['dataset']}_{self.config['blackbox_encoder']}_{self.config['surrogate_encoder']}"
        attacker_path = f"{save_dir}/" + \
            f"attacler_{self.config['dataset']}_{self.config['blackbox_encoder']}_{self.config['surrogate_encoder']}"

        # Load the model from the path
        self.mapping.load_state_dict(torch.load(map_path))
        self.projection.load_state_dict(torch.load(proj_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            attacker_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_dir'])
        print(f"Loading model from: {save_dir}")
        print(f"Loading mapping from: {map_path}")
        print(f"Loading projection from: {proj_path}")
        print(f"Loading attacker from: {attacker_path}")


class SurrogateModel(nn.Module):
    '''Surrogate model is used to mimic the black box encoder'''

    def __init__(self, config, device, output_dim=None):
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = config['surrogate_encoder']
        self.encoder_mapping = {'bert': 'bert-base-uncased',
                                'sbert': 'sentence-transformers/all-mpnet-base-v2',
                                'albert': 'albert-base-v2', 
                                'xlnet': 'xlnet-base-cased',
                                'ernie': 'nghuyong/ernie-2.0-base-en',
                                'gpt2': 'gpt2',
                                'st5-base':'sentence-transformers/sentence-t5-base',
                                'st5-large':'sentence-transformers/sentence-t5-large',
                                'st5-xl':'sentence-transformers/sentence-t5-xl',
                                'st5-xxl':'sentence-transformers/sentence-t5-xxl',
                                'gtr-base':'sentence-transformers/gtr-t5-base',
                                'gtr-large':'sentence-transformers/gtr-t5-large',
                                'gtr-xl':'sentence-transformers/gtr-t5-xl',
                                'gtr-xxl':'sentence-transformers/gtr-t5-xxl',
                                'gte-small':'thenlper/gte-small',
                                'gte-base':'thenlper/gte-base',
                                'gte-large':'thenlper/gte-large',
                                'e5-small':'intfloat/e5-small-v2',
                                'e5-base':'intfloat/e5-base-v2',
                                'e5-large':'intfloat/e5-large-v2',
                                }
        if self.encoder != 'openai':
            self.model = SentenceTransformer( self.encoder_mapping[self.encoder],device=self.device)
            self.output_dim = self.model.get_sentence_embedding_dimension()

    def forward(self, documents):
        '''Forward the document to the surrogate model'''
        embeddings = self.model.encode(documents,
                                       convert_to_numpy=True,
                                       show_progress_bar=True,
                                       )
        return embeddings

    def encode(self, documents, doc_type):
        if self.encoder == 'openai':
            if doc_type == "private":
                try:
                    with open(f"/data1/emb_attack/processed_data/openai_emb_private_aug_{self.config['dataset']}.pkl", 'rb') as f:
                        emb = pickle.load(f)
                    print("Load exist private + augment documents surrogate(OpenAI) embeddings:", len(emb))
                except:
                    emb = get_document_embs(documents, self.encoder)
                    with open(f"/data1/emb_attack/processed_data/openai_emb_private_aug_{self.config['dataset']}.pkl", 'wb') as f:
                        pickle.dump(emb, f)
                    print("Produce new private + augment documents surrogate(OpenAI) embeddings:", len(emb))
            elif doc_type == "external":
                try:
                    with open(f"/data1/emb_attack/processed_data/openai_emb_external_{self.config['external_dataset']}.pkl", 'rb') as f:
                        emb = pickle.load(f)
                    print("Load exist external documents surrogate(OpenAI) embeddings:", len(emb))
                except:
                    emb = get_document_embs(documents, self.encoder)
                    with open(f"/data1/emb_attack/processed_data/openai_emb_external_{self.config['external_dataset']}.pkl", 'wb') as f:
                        pickle.dump(emb, f)
                    print("Produce new external documents surrogate(OpenAI) embeddings:", len(emb))

            self.output_dim = emb.shape[1]
            return emb
        
        return self.forward(documents)

    def fit(self, data_loader):
        '''Training to make surrogate model act like black box encoder'''
        print(f"Training surrogate model on {self.encoder} encoder")
        self.train()
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters())
        for epoch in range(self.config['surrogate_epoch']):
            train_loss = 0
            for _, (corpus, embs) in enumerate(data_loader):
                embs = embs.to(self.device)     # [batch_size, embedding_dim]
                output = self.forward(corpus)  # [batch_size, embedding_dim]
                loss = loss_func(embs, output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(
                f"Epoch[{epoch+1}/{self.config['surrogate_epoch']}] \
                    Encoder loss:{train_loss / len(data_loader)}\n")
