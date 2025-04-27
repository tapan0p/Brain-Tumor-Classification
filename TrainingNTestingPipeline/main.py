import argparse
from brainTumorClassificationModel import BrainTumorClassifier
from pipeline import Pipeline



def main(args):
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    model = BrainTumorClassifier(num_classes=4)
    pipeline = Pipeline()
    train_loader,val_loader,test_loader = pipeline.load_data(train_data_path=args.train_data_path,test_data_path=args.test_data_path,batch_size=args.b)
    model = pipeline.train_model(model,train_loader,val_loader,args.epochs,args.lr)
    pipeline.test_model(model,test_loader,class_names)
    pipeline.save_model(model,model_name="brain_tumor_classifier")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Classification.")
    
    parser.add_argument("--train_data_path",type=str,required=True,help="Path to training dataset.")
    parser.add_argument("--test_data_path",type=str,required=True,help="Path to testing dataset.")
    parser.add_argument("--lr",type=float,default=1e-4,help="Learning rate")
    parser.add_argument("--b",type=int,default=32,help="Batch size")
    parser.add_argument("--epochs",type=int,default=35,help="Number of epochs to run")

    args = parser.parse_args()
    main(args)

