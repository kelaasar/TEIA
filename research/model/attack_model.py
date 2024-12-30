'''Collect the attacking model and surrogate model'''
import json
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import trange
from pathlib import Path
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, \
    get_linear_schedule_with_warmup
from research.utils.generation import top_filtering
from research.utils.optim import SequenceCrossEntropyLoss, prepare_optimizer
from research.utils.eval import calculate_metrics


class LinearProjection(nn.Module):
    '''Linear projection layer to project the embedding to the same dim as LM'''

    def __init__(self, in_num, out_num):
        super().__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, embs):
        '''Forward the embedding to the projection layer'''
        return self.fc1(embs)


class LLMAttackModel():
    '''LM based attack model'''

    def __init__(self, config, emb_dim, device):
        self.config = config
        self.emb_dim = emb_dim
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_dir']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_dir'])
        self.projection = LinearProjection(
            in_num=self.emb_dim, out_num=self.model.config.hidden_size).to(self.device)
        self.criterion = SequenceCrossEntropyLoss()
        self.optimizer = prepare_optimizer(self.model)
        self.optimizer.add_param_group(
            {'params': self.projection.parameters()})

    def fit(self, train_loader, val_loader):
        '''Training the attack model'''
        num_gradients_accumulation = 1
        num_train_optimization_steps = len(
            train_loader) * self.config['num_epochs'] // num_gradients_accumulation
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=100,
                                                    num_training_steps=num_train_optimization_steps)

        # process to obtain the embeddings
        for epoch in trange(self.config['num_epochs']):
            for _, (batch_text, embeddings) in enumerate(train_loader):
                embeddings = self.projection(embeddings.to(self.device))
                train_loss, train_perplexity = self.train_on_batch(
                    embeddings=embeddings, text=batch_text)
                train_loss.backward()
                self.optimizer.step()
                scheduler.step()
                # make sure no grad for GPT optimizer
                self.optimizer.zero_grad()

            # Evaluate on validation set
            if (epoch + 1) % self.config['eval_per_epochs'] == 0:
                result = self.eval_on_batch(val_loader)
                print(
                    f"Epoch:[{epoch+1}/{self.config['num_epochs']}], Train_loss: {train_loss.item()}, Train_perplexity: {train_perplexity}")
                print(result)
                if not self.config['testing']:
                    wandb.log(result)

                self.save_models(epoch)

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
        perplexity = np.exp(loss.item())
        return loss, perplexity

    def eval_on_batch(self, dataloader):
        '''Evaluate on a batch of data'''
        sent_dict = {}
        sent_dict['gt'] = []
        sent_dict['pred'] = []
        with torch.no_grad():
            for _, (batch_text, embeddings) in enumerate(dataloader):

                embeddings = embeddings.to(self.device)
                embeddings = self.projection(embeddings)
                sent_list, gt_list = self.inference_on_batch(
                    embeddings=embeddings, sentences=batch_text, config=self.config)
                sent_dict['pred'].extend(sent_list)
                sent_dict['gt'].extend(gt_list)

        # Evaluate the result
        return calculate_metrics(sent_dict['pred'], sent_dict['gt'])

    def predict(self, dataloader, config, epoch):
        '''Predict on a batch of data'''
        sent_dict = {}
        sent_dict['gt'] = []
        sent_dict['pred'] = []
        self.load_models(epoch)
        with torch.no_grad():
            for _, (batch_text, embeddings) in enumerate(dataloader):

                embeddings = embeddings.to(self.device)
                embeddings = self.projection(embeddings)
                sent_list, gt_list = self.inference_on_batch(
                    embeddings=embeddings, sentences=batch_text, config=config)
                sent_dict['pred'].extend(sent_list)
                sent_dict['gt'].extend(gt_list)

            # Evaluate the result
            result = calculate_metrics(sent_dict['pred'], sent_dict['gt'])
            print(result)

            # Save generated sentences to file
            output_dir = Path(config['exp_name'])
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / \
                Path(
                    f"{config['model_dir'].replace('/', '-')}_{config['train_ratio']}_{config['dataset']}.log")
            print(f"Saving result to: {output_path}")
            with open(output_path, 'w', encoding='UTF-8') as output_file:
                json.dump(sent_dict, output_file, indent=4)

    def inference_on_batch(self, embeddings, sentences, config):
        '''Evaluate on a batch of data'''
        decode_method = config['decode']
        embeddings = embeddings.to(self.device)
        # print(f'embeddings:{embeddings.size()}')
        sent_list = []
        gt_list = sentences
        for _, hidden in enumerate(embeddings):
            inputs_embeds = hidden
            if decode_method == 'sampling':
                sentence = self.generate_sentence(
                    hidden_embedding=inputs_embeds)
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
            logits = top_filtering(logits, top_k=top_k, top_p=top_p)

            probs = torch.softmax(logits, dim=-1)

            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == eos[0]:
                break
            sent.append(prev_word)

        output = self.tokenizer.decode(sent)

        return output

    def save_models(self, epoch):
        '''Save the model to the path'''
        save_dir = Path(
            f"{self.config['exp_name']}/{self.config['train_ratio']}/{epoch}/")
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        proj_path = f"{save_dir}/" + 'projection_gpt2_large_' + \
            self.config['dataset'] + '_' + self.config['blackbox_encoder']
        torch.save(self.projection.state_dict(), proj_path)
        save_path = f"{save_dir}/" + 'attacker_gpt2_large_' + \
            self.config['dataset'] + '_' + self.config['blackbox_encoder']

        print(f"Saving model to: {save_path}")
        self.model.save_pretrained(save_path)

    def load_models(self, epoch):
        '''Load the model from the path'''
        save_dir = Path(
            f"{self.config['exp_name']}/{self.config['train_ratio']}/{epoch}/")
        proj_path = f"{save_dir}/" + 'projection_gpt2_large_' + \
            self.config['dataset'] + '_' + self.config['blackbox_encoder']
        self.projection = LinearProjection(
            in_num=self.emb_dim, out_num=self.model.config.hidden_size)
        self.projection.load_state_dict(torch.load(proj_path))
        self.projection.to(self.device)

        attacker_path = f"{save_dir}/" + 'attacker_gpt2_large_' + \
            self.config['dataset'] + '_' + self.config['blackbox_encoder']

        print(f"Loading model from: {attacker_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            attacker_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_dir'])


class SurrogateModel(nn.Module):
    '''Surrogate model is used to mimic the black box encoder'''

    def __init__(self, config, device, output_dim=None):
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = config['surrogate_encoder']
        self.encoder_mapping = {'bert': 'bert-base-uncased',
                                'mpnet': 'sentence-transformers/all-mpnet-base-v2',
                                'sbert': 'sentence-transformers/all-mpnet-base-v2',
                                'albert': 'albert-base-v2', 'xlnet': 'xlnet-base-cased',
                                'ernie': 'nghuyong/ernie-2.0-base-en', 'gpt2': 'gpt2'}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.encoder_mapping[self.encoder])
        self.model = AutoModel.from_pretrained(
            self.encoder_mapping[self.encoder]).to(self.device)
        if output_dim is not None:
            self.fc_layer = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, output_dim),
            ).to(self.device)
        else:
            self.fc_layer = None

        # Handle for different encoder
        if self.encoder == 'gpt2':
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, documents):
        '''Forward the document to the surrogate model'''
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True,
                                truncation=True).to(self.device)
        embedding = self.model(**inputs).last_hidden_state[:, 0, :]
        if self.fc_layer is not None:
            embedding = self.fc_layer(embedding)
        return embedding

    def encode(self, documents):
        '''Encode the document to document embedding'''
        doc_embs = []
        for sents in documents:
            embeddings = self.forward(sents)
            doc_embs.append(embeddings.detach().cpu().numpy()[0])

        return np.array(doc_embs)

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
