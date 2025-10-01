from research.config.base_config import parse_argument
from research.data.data_prepare import prepare_pivot_data, prepare_adv_additional_data, \
    prepare_geia_data, prepare_external_data, load_prepared_dataset, load_augmented_data
from research.utils.toolbox import get_free_gpu, same_seed, create_save_path
from research.utils.eval import eval_generation
from research.model.adv_attack_model import SurrogateModel, LLMAttackModel
from torch.utils.data import DataLoader
import wandb
import json
import torch


def get_best_step(val_result):
    """Get the best step based on validation results"""
    if not val_result:
        return 0
    
    best_step = 0
    best_score = 0
    
    for i, score in enumerate(val_result):
        # Handle different possible formats
        if isinstance(score, (list, tuple)) and len(score) > 0:
            current_score = score[0]
        elif isinstance(score, dict) and 'embed_similarity' in score:
            current_score = score['embed_similarity']
        elif isinstance(score, (int, float)):
            current_score = score
        else:
            print(f"Warning: Unexpected score format at step {i}: {score}")
            continue
            
        if current_score > best_score:
            best_score = current_score
            best_step = i
    
    return best_step


def eval_best_result(config, val_result):
    """Evaluate the best result"""
    if not val_result:
        print("📊 No validation results available")
        return
        
    best_step = get_best_step(val_result)
    print(f"📊 Best validation result at step {best_step}: {val_result[best_step]}")
    
    # Log best result to wandb if not testing
    if not config['testing']:
        best_result = val_result[best_step]
        if isinstance(best_result, (list, tuple)) and len(best_result) > 0:
            best_score = best_result[0]
        elif isinstance(best_result, dict) and 'embed_similarity' in best_result:
            best_score = best_result['embed_similarity']
        elif isinstance(best_result, (int, float)):
            best_score = best_result
        else:
            best_score = 0
            
        wandb.log({
            'best_embed_similarity': best_score,
            'best_step': best_step
        })


if __name__ == '__main__':
    # Initialize settings
    same_seed(123)
    args = parse_argument()
    config = vars(args)
    device = get_free_gpu()

    if not config['testing']:
        # Only monitor when not testing
        wandb.init(project=f"{config['project_name']}",
                   name=f"{config['training_size']}_{config['exp_name']}_{config['blackbox_encoder']}", config=config)
    else:
        # Mini batch config for testing
        config['exp_name'] = 'test'
        config['training_size'] = 100
        config['num_epochs'] = 2
        config['eval_per_epochs'] = 1
        config['surrogate_epoch'] = 5
        print("🧪 Running in test mode with reduced parameters")

    # Show configuration
    print(f"🚀 Starting TEIA training")
    print(f"📊 Project: {config['project_name']}")
    print(f"🔬 Experiment: {config['exp_name']}")
    print(f"🤖 Model: {config['model_dir']}")
    print(f"📚 Dataset: {config['dataset']}")
    print(f"🎯 Training size: {config['training_size']}")
    print(f"🔄 Epochs: {config['num_epochs']}")
    print(f"⚙️  Enhanced features:")
    print(f"   - Deep projection: {config.get('use_deep_projection', False)}")
    print(f"   - Embedding consistency weight: {config.get('embedding_consistency_weight', 0.0)}")
    print(f"   - Surrogate epochs: {config['surrogate_epoch']}")
    print(f"🖥️  Device: {device}")

    # Create save path (step 0 for initial setup)
    model_save_path = create_save_path(config, 0)
    print(f"💾 Model save path: {model_save_path}")

    # Load dataset
    print("\n📚 Loading datasets...")
    train_dataset, val_dataset = load_prepared_dataset(config)
    external_dataset = prepare_external_data(config)
    
    # Get private embedding dimension from the first sample
    private_emb_dim = train_dataset[0][1].shape[0]
    
    print(f"✅ Dataset loaded - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    print(f"🎯 Private embedding dimension: {private_emb_dim}")

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Train surrogate model
    print(f"\n🔧 Training surrogate model for {config['surrogate_epoch']} epochs...")
    surrogate = SurrogateModel(config, device)
    surrogate.fit(train_loader)

    # Prepare pivot data
    print("\n🔄 Preparing pivot data...")
    pivot_dataset = prepare_pivot_data(train_dataset, surrogate)
    surrogate_emb_dim = surrogate.output_dim
    print(f"✅ Surrogate embedding dimension: {surrogate_emb_dim}")

    # Prepare training data
    if not config['geia']:
        # Prepare new train loader that use additional data from surrogate model
        print(f"🚀 Using additional data from adversarial training")
        final_train_loader = prepare_adv_additional_data(
            pivot_dataset, external_dataset, surrogate, config)
    else:
        print(f"🔧 Using GEIA data")
        final_train_loader = prepare_geia_data(pivot_dataset, config)
    
    # Clean up surrogate model to free memory
    del surrogate
    torch.cuda.empty_cache()

    # Train attack model
    print(f"\n🎯 Training LLM attack model...")
    llm_attacker = LLMAttackModel(
        config, private_emb_dim, surrogate_emb_dim, device)
    val_result = llm_attacker.fit(final_train_loader, val_loader)
    
    # Clean up attack model
    del llm_attacker
    torch.cuda.empty_cache()

    # Evaluate best result
    print(f"\n📊 Evaluating best results...")
    eval_best_result(config, val_result)

    if not config['testing']:
        wandb.finish()
    
    print(f"\n✅ Training completed!")
    print(f"💾 Models saved to: {model_save_path}")