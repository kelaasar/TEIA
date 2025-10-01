#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.utils.generation import top_filtering
import torch.nn as nn


class LinearProjection(nn.Module):
    '''Compatible projection layer for loading saved models'''
    
    def __init__(self, in_num, out_num):
        super().__init__()
        # Use fc1 to match saved model structure
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, embs):
        '''Forward the embedding to the projection layer'''
        projected = self.fc1(embs)
        return torch.clamp(projected, min=-1e6, max=1e6)


# =============================================================================
# MODEL CONFIGURATION - CHANGE THESE TO SWITCH BETWEEN DIFFERENT TRAINED MODELS
# =============================================================================

# Available trained models:
MODEL_CONFIGS = {
    # Original small model (epoch 4 performance)
    "old_small": {
        "exp_name": "full_run_1",
        "model_step": 1,  # Step 1 was best (epoch 2)
        "training_size": "8000",
        "base_model": "microsoft/DialoGPT-small",
        "description": "Original DialoGPT-small, 8K samples, epoch 2"
    },
    
    # New improved medium model 
    "new_medium": {
        "exp_name": "dialogpt_medium_32k_bs32_nepoch18", 
        "model_step": 2,  # Step 2 was best (epoch 3) 
        "training_size": "32000",
        "base_model": "microsoft/DialoGPT-medium",
        "description": "Improved DialoGPT-medium, 32K samples, epoch 3"
    }
}

# SELECT WHICH MODEL TO USE:
ACTIVE_MODEL = "new_medium"  # Change this to switch models

# =============================================================================


class TEIAQuerySystem:
    def __init__(self, model_config_name: str = None):
        """Initialize the TEIA query system"""
        # Use active model config if none specified
        config_name = model_config_name or ACTIVE_MODEL
        
        if config_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
        
        self.config = MODEL_CONFIGS[config_name]
        self.exp_name = self.config["exp_name"]
        self.model_step = self.config["model_step"]
        self.training_size = self.config["training_size"]
        self.base_model = self.config["base_model"]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        
        print(f"🎯 Using model config: {config_name}")
        print(f"📋 {self.config['description']}")

    def load_models(self):
        if self.models_loaded:
            return

        print(f"Loading models on {self.device}...")

        # Victim SBERT
        print("📚 Loading victim SBERT model...")
        self.victim_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device=self.device
        )

        # DialoGPT attack model
        print(f"🤖 Loading {self.base_model} attack model...")
        self.attack_model = AutoModelForCausalLM.from_pretrained(
            self.base_model
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Projection layer
        victim_emb_dim = self.victim_model.get_sentence_embedding_dimension()
        hidden_dim = self.attack_model.config.hidden_size
        self.projection = LinearProjection(in_num=victim_emb_dim, out_num=hidden_dim).to(self.device)

        # Load trained weights
        model_dir = Path(self.exp_name) / "32000" / str(self.model_step)

        attack_model_path = model_dir / "attacler_qnli_sbert_gtr-base"
        if attack_model_path.exists():
            # Use TEIA's approach - load model directly with from_pretrained
            try:
                self.attack_model = AutoModelForCausalLM.from_pretrained(
                    str(attack_model_path)
                ).to(self.device)
                print("✅ TEIA fine-tuned model loaded")
                    
            except Exception as e:
                print(f"⚠️  Could not load TEIA model: {e}")
                print("📋 Using base DialoGPT model")
                self.attack_model = AutoModelForCausalLM.from_pretrained(
                    'microsoft/DialoGPT-medium'
                ).to(self.device)
        else:
            print(f"⚠️  Warning: Could not find attack model at {attack_model_path}")
            print("📋 Using base DialoGPT model")

        projection_path = model_dir / "projection_qnli_sbert_gtr-base"
        if projection_path.exists():
            self.projection.load_state_dict(torch.load(projection_path, map_location=self.device))
            print("✅ Projection loaded")
        else:
            print(f"⚠️  Warning: Could not find projection at {projection_path}")

        # Eval mode
        self.attack_model.eval()
        self.projection.eval()
        self.models_loaded = True

        print(f"🎯 Ready! Using final model (step {self.model_step}, epoch {(self.model_step + 1) * 2}) on {self.device}")

    def embed_sentence(self, sentence: str):
        embedding = self.victim_model.encode([sentence], convert_to_tensor=True)
        return embedding[0]

    def invert_embedding(self, embedding: torch.Tensor, max_length: int = 50):
        with torch.no_grad():
            hidden_embedding = self.projection(embedding.unsqueeze(0)).squeeze(0)
            return self._generate_sentence(hidden_embedding, max_length)

    def _generate_sentence(self, hidden_embedding: torch.Tensor, max_length: int = 50):
        temperature = 0.9
        top_k, top_p = -1, 0.9
        eos = self.tokenizer.encode("<|endoftext|>")

        sent, past = [], None
        hidden_embedding = hidden_embedding.unsqueeze(0).unsqueeze(0)

        logits, past = self.attack_model(inputs_embeds=hidden_embedding, past_key_values=past, return_dict=False)
        for _ in range(max_length):
            logits = logits[:, -1, :] / temperature
            logits = top_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()
            if prev_word == eos[0]:
                break
            sent.append(prev_word)
            logits, past = self.attack_model(prev_input, past_key_values=past, return_dict=False)

        return self.tokenizer.decode(sent).replace('<|endoftext|>', '').strip()

    def query(self, sentence: str):
        print(f"\n🎯 Original: {sentence}")
        embedding = self.embed_sentence(sentence)
        inverted = self.invert_embedding(embedding)
        print(f"🔓 Inverted: {inverted}")

        # Calculate cosine similarity between original and inverted embeddings
        original_embedding = self.embed_sentence(sentence)
        inverted_embedding = self.embed_sentence(inverted)
        
        # Convert to numpy arrays and reshape for cosine_similarity
        orig_emb = original_embedding.cpu().numpy().reshape(1, -1)
        inv_emb = inverted_embedding.cpu().numpy().reshape(1, -1)
        
        similarity = cosine_similarity(orig_emb, inv_emb)[0][0]
        print(f"� Cosine similarity: {similarity:.3f}")

        return inverted

    def interactive_mode(self):
        print("\n🚀 TEIA Interactive Query Mode")
        print("=" * 60)
        print("Commands: 'quit'\n")
        while True:
            try:
                user_input = input("\n💬 Enter sentence: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                if not user_input:
                    print("⚠️ Please enter a sentence.")
                    continue

                if not self.models_loaded:
                    self.load_models()

                self.query(user_input)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    # The active model is configured at the top of this file in ACTIVE_MODEL
    # To switch models, change ACTIVE_MODEL or pass a different config name
    system = TEIAQuerySystem()  # Uses ACTIVE_MODEL by default
    # system = TEIAQuerySystem("old_small")  # Uncomment to use old model
    system.interactive_mode()


if __name__ == "__main__":
    main()
