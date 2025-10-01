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
import torch.nn.functional as F


class LinearProjection(nn.Module):
    '''Enhanced projection layer with multiple architecture options'''

    def __init__(self, in_num, out_num, use_deep=False, dropout_rate=0.1, architecture='simple'):
        super().__init__()
        self.use_deep = use_deep
        self.architecture = architecture
        
        if architecture == 'residual':
            # Residual connection network
            hidden_dim = max(in_num, out_num)
            self.projection = ResidualProjection(in_num, hidden_dim, out_num, dropout_rate)
        elif architecture == 'transformer':
            # Transformer-based projection
            self.projection = TransformerProjection(in_num, out_num, dropout_rate)
        elif architecture == 'deep':
            # Multi-layer projection for better information preservation
            hidden_dim = max(in_num, out_num)
            self.projection = nn.Sequential(
                nn.Linear(in_num, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, out_num)
            )
        else:
            # Simple linear projection (original)
            self.projection = nn.Linear(in_num, out_num)

    def forward(self, embs):
        '''Forward the embedding to the projection layer'''
        projected = self.projection(embs)
        return torch.clamp(projected, min=-1e6, max=1e6)


class ResidualProjection(nn.Module):
    '''Residual projection network for better gradient flow'''
    
    def __init__(self, in_num, hidden_dim, out_num, dropout_rate=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_num, hidden_dim)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(3)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_num)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm(x)
        
        for block in self.residual_blocks:
            x = block(x)
            
        return self.output_proj(x)


class ResidualBlock(nn.Module):
    '''Individual residual block'''
    
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return self.norm2(x + residual)


class TransformerProjection(nn.Module):
    '''Transformer-based projection using attention mechanisms'''
    
    def __init__(self, in_num, out_num, dropout_rate=0.1):
        super().__init__()
        hidden_dim = max(in_num, out_num)
        
        self.input_proj = nn.Linear(in_num, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.output_proj = nn.Linear(hidden_dim, out_num)
        
    def forward(self, x):
        # Add sequence dimension for transformer
        x = self.input_proj(x).unsqueeze(1)  # [batch, 1, hidden]
        x = self.transformer(x)
        x = x.squeeze(1)  # [batch, hidden]
        return self.output_proj(x)


class MappingNetwork(nn.Module):
    '''Enhanced mapping networks with multiple architecture options'''

    def __init__(self, input_dim, output_dim, architecture='simple'):
        super().__init__()
        self.architecture = architecture
        
        if architecture == 'deep':
            # Deeper mapping network
            hidden1 = max(input_dim, output_dim)
            hidden2 = (input_dim + output_dim) // 2
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.LayerNorm(hidden1),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden1, hidden2),
                nn.LayerNorm(hidden2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden2, output_dim)
            )
        elif architecture == 'residual':
            # Residual mapping network
            self.net = ResidualMapping(input_dim, output_dim)
        else:
            # Simple mapping network (original)
            self.net = nn.Sequential(
                nn.Linear(input_dim, (input_dim + output_dim) // 2),
                nn.ReLU(),
                nn.Linear((input_dim + output_dim) // 2, output_dim)
            )

    def forward(self, embs):
        '''Forward the embedding to the mapping network'''
        return self.net(embs.float())


class ResidualMapping(nn.Module):
    '''Residual mapping network for better information preservation'''
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = max(input_dim, output_dim)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(2)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_proj(x)


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


class ContrastiveLoss(nn.Module):
    '''Contrastive loss for better embedding mapping'''
    
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, emb1, emb2, labels=None):
        '''
        emb1, emb2: embeddings to compare
        labels: 1 for positive pairs, 0 for negative pairs
        '''
        if labels is None:
            # Assume all pairs are positive (same semantic content)
            labels = torch.ones(emb1.size(0), device=emb1.device)
            
        distances = F.pairwise_distance(emb1, emb2)
        
        positive_loss = labels * torch.pow(distances, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        return torch.mean(positive_loss + negative_loss)


class TripletLoss(nn.Module):
    '''Triplet loss for embedding alignment'''
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        '''
        anchor: original embeddings
        positive: mapped embeddings (should be close to anchor)
        negative: random embeddings (should be far from anchor)
        '''
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)


class InfoNCELoss(nn.Module):
    '''InfoNCE loss for contrastive learning'''
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query, positive, negatives):
        '''
        query: anchor embeddings
        positive: positive embeddings 
        negatives: negative embeddings
        '''
        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # Compute similarities
        pos_sim = torch.sum(query * positive, dim=-1) / self.temperature
        neg_sim = torch.matmul(query, negatives.transpose(-2, -1)) / self.temperature
        
        # Compute InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
        return F.cross_entropy(logits, labels)


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
        
        # Enhanced projection with architecture selection
        proj_arch = config.get('projection_architecture', 'simple')
        self.projection = LinearProjection(
            in_num=self.emb_dim, out_num=self.model.config.hidden_size, 
            use_deep=config.get('use_deep_projection', False),
            architecture=proj_arch).to(self.device)
            
        self.criterion = SequenceCrossEntropyLoss()
        self.optimizer = prepare_optimizer(self.model)
        self.optimizer.add_param_group(
            {'params': self.projection.parameters()}
        )
        
        # Add embedding consistency loss
        self.victim_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2', device=self.device)
        self.embedding_consistency_weight = config.get('embedding_consistency_weight', 0.5)

        # Pivot training
        self.mse_loss = torch.nn.MSELoss()
        self.pairwise_pivot_loss = PairwisePivotLoss()

        # Adversarial training with enhanced mapping
        self.bce_loss = torch.nn.BCELoss()
        mapping_arch = config.get('mapping_architecture', 'simple')
        self.mapping = MappingNetwork(emb2_dim, emb_dim, architecture=mapping_arch).to(self.device)
        self.discriminator = Discriminator(emb_dim).to(self.device)
        
        # Advanced loss functions
        self.contrastive_loss = ContrastiveLoss(margin=1.0, temperature=0.1)
        self.triplet_loss = TripletLoss(margin=1.0)
        self.infonce_loss = InfoNCELoss(temperature=0.1)
        self.loss_type = config.get('loss_type', 'standard')  # 'standard', 'contrastive', 'triplet', 'infonce'
        
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
                    
                    # Add embedding consistency loss during GEIA training
                    embed_consistency_loss = self.compute_embedding_consistency_loss(emb1, text1)
                    loss = train_loss + self.embedding_consistency_weight * embed_consistency_loss

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
                result['epoch'] = epoch + 1
                print(result)
                if not self.config['testing']:
                    wandb.log(result, step=step)

                result_score.append(result)
                self.save_models(step)
                step += 1

        return result_score

    def compute_embedding_consistency_loss(self, embeddings, texts):
        '''Compute embedding consistency loss by comparing reconstructed vs original embeddings'''
        with torch.no_grad():
            # Generate sentences from embeddings
            projected_embs = self.projection(embeddings)
            reconstructed_texts = []
            
            for hidden in projected_embs:
                # Quick reconstruction without full generation
                recon_text = self.generate_sentence_fast(hidden.unsqueeze(0))
                reconstructed_texts.append(recon_text)
            
            # Get embeddings of reconstructed texts
            if reconstructed_texts and any(text.strip() for text in reconstructed_texts):
                recon_embeddings = torch.tensor(
                    self.victim_model.encode(reconstructed_texts, convert_to_tensor=False),
                    device=self.device, dtype=embeddings.dtype
                )
                
                # Cosine similarity loss (encourage higher similarity)
                cos_sim = torch.nn.functional.cosine_similarity(embeddings, recon_embeddings, dim=1)
                # Convert to loss (1 - similarity, so minimizing increases similarity)
                consistency_loss = 1.0 - cos_sim.mean()
                return consistency_loss
            else:
                return torch.tensor(0.0, device=self.device)
    
    def generate_sentence_fast(self, hidden_embedding, max_length=20):
        '''Fast sentence generation for consistency loss (shorter sequences)'''
        temperature = 0.7  # Lower temperature for more focused generation
        sent = []
        past = None
        eos = self.tokenizer.encode("<|endoftext|>")
        
        hidden_embedding = hidden_embedding.unsqueeze(0)  # [1,1,embed_dim]
        logits, past = self.model(inputs_embeds=hidden_embedding,
                                  past_key_values=past, return_dict=False)
        logits = logits[:, -1, :] / temperature
        logits = torch.clamp(logits, min=-1e9, max=1e9)
        probs = torch.softmax(logits, dim=-1)

        prev_input = torch.multinomial(probs, num_samples=1)
        prev_word = prev_input.item()
        sent.append(prev_word)

        for _ in range(max_length):
            logits, past = self.model(prev_input, past_key_values=past, return_dict=False)
            logits = logits[:, -1, :] / temperature
            logits = torch.clamp(logits, min=-1e9, max=1e9)
            probs = torch.softmax(logits, dim=-1)

            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == eos[0]:
                break
            sent.append(prev_word)

        return self.tokenizer.decode(sent).replace('<|endoftext|>', '').strip()

    def pivot_on_batch(self, emb1, emb2):
        '''Calculate pivot losses with advanced loss functions'''
        if self.loss_type == 'contrastive':
            # Use contrastive loss (positive pairs)
            labels = torch.ones(emb1.size(0), device=self.device)
            contrastive = self.contrastive_loss(emb1, emb2, labels)
            pairwise_loss = self.pairwise_pivot_loss(emb1, emb2)
            return contrastive + 0.5 * pairwise_loss
            
        elif self.loss_type == 'triplet':
            # Create negative samples by shuffling embeddings
            batch_size = emb1.size(0)
            indices = torch.randperm(batch_size, device=self.device)
            negative = emb1[indices]
            
            triplet = self.triplet_loss(emb1, emb2, negative)
            pairwise_loss = self.pairwise_pivot_loss(emb1, emb2)
            return triplet + 0.3 * pairwise_loss
            
        elif self.loss_type == 'infonce':
            # Use InfoNCE loss
            batch_size = emb1.size(0)
            if batch_size > 1:
                # Create negatives from other samples in batch
                negatives = []
                for i in range(batch_size):
                    neg_indices = [j for j in range(batch_size) if j != i]
                    if neg_indices:
                        neg_sample = emb1[neg_indices]
                        negatives.append(neg_sample)
                
                if negatives:
                    # Pad negatives to same size
                    max_negs = max(neg.size(0) for neg in negatives)
                    padded_negs = []
                    for neg in negatives:
                        if neg.size(0) < max_negs:
                            padding = max_negs - neg.size(0)
                            pad_neg = torch.cat([neg, neg[:padding]], dim=0)
                            padded_negs.append(pad_neg)
                        else:
                            padded_negs.append(neg[:max_negs])
                    
                    negatives_tensor = torch.stack(padded_negs, dim=0)
                    infonce = self.infonce_loss(emb1, emb2, negatives_tensor)
                    return infonce + 0.2 * self.mse_loss(emb1, emb2)
        
        # Standard loss (fallback)
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
        '''Generate sentence using LLM with optimized parameters'''
        # Better generation parameters for higher embedding similarity
        temperature = 0.6  # Lower temperature for more focused generation
        top_k = -1
        top_p = 0.8  # More restrictive nucleus sampling
        max_length = 35  # Shorter sequences tend to have better similarity
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

        for _ in range(max_length):
            # Use original model for generation to avoid DataParallel issues
            model_for_generation = self._original_model if hasattr(self, '_original_model') else self.model
            logits, past = model_for_generation(
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
            self.model = SentenceTransformer(self.encoder_mapping[self.encoder], device=self.device)
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
        '''Enhanced training to make surrogate model act like black box encoder'''
        print(f"Training enhanced surrogate model on {self.encoder} encoder")
        self.train()
        
        # Enhanced loss functions for better embedding quality
        mse_loss = torch.nn.MSELoss()
        cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0)
        contrastive_loss = ContrastiveLoss(margin=1.0, temperature=0.1)
        
        # Better optimizer with learning rate scheduling
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['surrogate_epoch'])
        
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(self.config['surrogate_epoch']):
            train_loss = 0
            mse_total = 0
            cosine_total = 0
            
            for batch_idx, (corpus, embs) in enumerate(data_loader):
                embs = embs.to(self.device)     # [batch_size, embedding_dim]
                output = self.forward(corpus)  # [batch_size, embedding_dim]
                
                # Convert output to tensor if it's numpy array
                if isinstance(output, np.ndarray):
                    output = torch.tensor(output, dtype=embs.dtype, device=self.device, requires_grad=True)
                
                # Normalize embeddings for better cosine similarity
                embs_norm = torch.nn.functional.normalize(embs, p=2, dim=1)
                output_norm = torch.nn.functional.normalize(output, p=2, dim=1)
                
                # Multi-objective loss: MSE + Cosine + Contrastive + L1 regularization
                mse = mse_loss(output_norm, embs_norm)  # Normalized MSE
                
                # Cosine similarity loss (want high similarity = low loss)
                target_labels = torch.ones(embs.size(0), device=self.device)
                cosine = cosine_loss(output_norm, embs_norm, target_labels)
                
                # Contrastive loss for better alignment
                contrastive = contrastive_loss(output_norm, embs_norm, target_labels)
                
                # L1 regularization to prevent overfitting
                l1_reg = sum(param.abs().sum() for param in self.parameters()) * 1e-6
                
                # Combined loss with optimized weights
                loss = 0.5 * mse + 0.2 * cosine + 0.3 * contrastive + l1_reg
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                mse_total += mse.item()
                cosine_total += cosine.item()
            
            scheduler.step()
            avg_loss = train_loss / len(data_loader)
            avg_mse = mse_total / len(data_loader)
            avg_cosine = cosine_total / len(data_loader)
            
            print(f"Epoch[{epoch+1}/{self.config['surrogate_epoch']}] "
                  f"Total Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, "
                  f"Cosine: {avg_cosine:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Early stopping based on loss improvement
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model state
                self.best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch > self.config['surrogate_epoch'] // 2:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement")
                    # Load best state
                    if hasattr(self, 'best_state'):
                        self.load_state_dict(self.best_state)
                    break
        
        print(f"✅ Enhanced surrogate training complete. Best loss: {best_loss:.6f}")
        
        # Evaluate surrogate quality on training data
        self.evaluate_surrogate_quality(data_loader)
    
    def evaluate_surrogate_quality(self, data_loader, num_samples=100):
        '''Evaluate surrogate model quality with similarity metrics'''
        self.eval()
        similarities = []
        mse_scores = []
        
        with torch.no_grad():
            sample_count = 0
            for _, (corpus, embs) in enumerate(data_loader):
                if sample_count >= num_samples:
                    break
                    
                embs = embs.to(self.device)
                output = self.forward(corpus)
                
                if isinstance(output, np.ndarray):
                    output = torch.tensor(output, dtype=embs.dtype, device=self.device)
                
                # Normalize for cosine similarity
                embs_norm = torch.nn.functional.normalize(embs, p=2, dim=1)
                output_norm = torch.nn.functional.normalize(output, p=2, dim=1)
                
                # Calculate cosine similarities
                cos_sim = torch.nn.functional.cosine_similarity(embs_norm, output_norm, dim=1)
                similarities.extend(cos_sim.cpu().tolist())
                
                # Calculate MSE
                mse = torch.nn.functional.mse_loss(embs_norm, output_norm, reduction='none').mean(dim=1)
                mse_scores.extend(mse.cpu().tolist())
                
                sample_count += embs.size(0)
        
        avg_similarity = np.mean(similarities)
        avg_mse = np.mean(mse_scores)
        
        print(f"📊 Surrogate Quality Metrics:")
        print(f"   Average Cosine Similarity: {avg_similarity:.4f}")
        print(f"   Average MSE: {avg_mse:.6f}")
        print(f"   Similarity Std: {np.std(similarities):.4f}")
        
        if avg_similarity > 0.85:
            print("🟢 Excellent surrogate quality!")
        elif avg_similarity > 0.75:
            print("🟡 Good surrogate quality")
        elif avg_similarity > 0.65:
            print("🟠 Moderate surrogate quality - consider more training")
        else:
            print("🔴 Poor surrogate quality - needs improvement")
        
        self.train()  # Switch back to training mode
