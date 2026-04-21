from ultralytics import YOLO
from pathlib import Path
import yaml
import torch

 
root = Path(__file__).parent / "spheres_and_cubes" 
classes = {0: "cube", 1: "neither", 2: "sphere"}


config = {
    "path" : str(root.absolute()),
    "train" : str((root/"images/train").absolute()),
    "val" : str((root/"images/val").absolute()),
    "nc" : len(classes),
    "names" : classes
}

with open(root / "dataset.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config, f,allow_unicode=True)

if __name__ == '__main__':
    size = "n" 
    model= YOLO("yolo26n.pt")

    result = model.train(
        data=str(root / "dataset.yaml"),
        imgsz=640,
        batch=16,
        workers=2,

        epochs = 60,
        patience = 5,
        optimizer = "AdamW",
        lr0 = 0.001,
        warmup_epochs = 5,
        cos_lr = True,

        dropout = 0.3,

        hsv_h=0.015, 
        hsv_s=0.7,
        hsv_v=0.4,
        flipud = 0.0,
        fliplr = 0.5,
        mosaic = 1.0,
        degrees = 5.0,
        scale = 0.5,
        translate = 0.1,

        conf = 0.01,
        iou = 0.7,

        project = "figures",
        name = "yolo",
        save = True,
        save_period = 5,
        device = 0 if torch.cuda.is_available() else "cpu",

        verbose = True,
        plots =True,
        val = True,
        close_mosaic = 8,
        amp = True
    )
    print("Done")
    print(result.save_dir)