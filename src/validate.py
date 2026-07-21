from configs import config
from src.engine.validator import run_validation

def main() -> None:
    device = config.DEVICE
    checkpoint_path = config.MODEL_SAVE_PATH
    
    try:
        run_validation(checkpoint_path, device)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
