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

from research.model.adv_attack_model import LinearProjection
from research.utils.generation import top_filtering


class TEIAQuerySystem:
    def __init__(self, exp_name: str = "full_run_1", model_step: int = 1):
        """Initialize the TEIA query system"""
        self.exp_name = exp_name
        self.model_step = model_step
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False

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
        print("🤖 Loading DialoGPT attack model...")
        self.attack_model = AutoModelForCausalLM.from_pretrained(
            'microsoft/DialoGPT-small'
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')

        # Projection layer
        victim_emb_dim = self.victim_model.get_sentence_embedding_dimension()
        hidden_dim = self.attack_model.config.hidden_size
        self.projection = LinearProjection(in_num=victim_emb_dim, out_num=hidden_dim).to(self.device)

        # Load trained weights
        model_dir = Path(self.exp_name) / "8000" / str(self.model_step)

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
                    'microsoft/DialoGPT-small'
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

    def show_training_examples(self, num_examples: int = 5):
        result_file = Path(f"{self.exp_name}/0.1_{self.model_step}.log")
        if not result_file.exists():
            print(f"❌ Could not find result file: {result_file}")
            return

        print(f"🔍 Loading results from epoch {(self.model_step + 1) * 2}...")
        with open(result_file, 'r') as f:
            data = json.load(f)

        gt_sentences = data['gt'][:num_examples]
        pred_sentences = data['pred'][:num_examples]

        print(f"\n📊 Showing {num_examples} examples:")
        print("=" * 60)
        for i, (gt, pred) in enumerate(zip(gt_sentences, pred_sentences), 1):
            print(f"[{i}] Original: {gt}")
            print(f"    Inverted: {pred}")

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
        print("Commands: 'examples' | 'quit'\n")
        while True:
            try:
                user_input = input("\n💬 Enter sentence: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                if user_input.lower() in ['examples', 'ex']:
                    self.show_training_examples()
                    continue
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
    system = TEIAQuerySystem("full_run_1")  # Uses default model_step=11 (final model)
    system.interactive_mode()


if __name__ == "__main__":
    main()
