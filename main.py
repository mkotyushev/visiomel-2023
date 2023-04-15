import importlib
import torch
import yaml
from pytorch_lightning import seed_everything    


CONFIG_PATH = 'run/configs/fake_config.yaml'


def get_class(class_path):
    module_path = '.'.join(class_path.split('.')[:-1])
    class_name = class_path.split('.')[-1]
    return getattr(importlib.import_module(module_path), class_name)


def main():
    seed_everything(0)

    # Get classes
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    ModelClass = get_class(config['model']['class_path'])
    DatamoduleClass = get_class(config['data']['class_path'])
    PredictionWriterClass = get_class(config['trainer']['callbacks'][0]['class_path'])

    # Load model
    model = ModelClass.load_from_checkpoint('final.ckpt')

    # Load data
    datamodule = DatamoduleClass()
    datamodule.setup('test')
    test_dataloader = datamodule.test_dataloader()

    # Load prediction writer
    prediction_writer = PredictionWriterClass('submission.csv')

    # Predict 
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            predictions.extend(model(batch))

    # Write predictions
    prediction_writer.write_on_epoch_end(None, None, predictions, None)


if __name__ == '__main__':
    main()
