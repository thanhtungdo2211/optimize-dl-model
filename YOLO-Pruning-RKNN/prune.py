from ultralytics import YOLO

model = YOLO('yolov8s.pt', verbose=True)

# from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
# weights, _ = attempt_load_one_weight('yolov8s.pt')

# model = DetectionModel(cfg="yolov8s.yaml")
# model.load(weights)


def prunetrain(train_epochs, prune_epochs=0, quick_pruning=True, prune_ratio=0.5, 
               prune_iterative_steps=1, data='coco.yaml', name='yolo11', imgsz=640, 
               batch=8, device=[0], sparse_training=False):
    if not quick_pruning:
        assert train_epochs > 0 and prune_epochs > 0, "Quick Pruning is not set. prune epochs must > 0."
        model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch, device=device, name=name, prune=False,
                    sparse_training=sparse_training)
        return model.train(data=data, epochs=prune_epochs, imgsz=imgsz, batch=batch, device=device, name=name, prune=True,
                           prune_ratio=prune_ratio, prune_iterative_steps=prune_iterative_steps)
    else:
        return model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch, device=device, 
                           name=name, prune=True, prune_ratio=prune_ratio, prune_iterative_steps=prune_iterative_steps)

# Normal Pruning
prunetrain(quick_pruning=True,       # Quick Pruning or not
           data='ultralytics/cfg/datasets/coco_test.yaml',          # Dataset config
           train_epochs=30,           # Epochs before pruning
           prune_epochs=30,           # Epochs after pruning 
           imgsz=640,                 # Input size
           batch=32,                   # Batch size
           device=[0],                # GPU devices
           name='yolov8s_pruning',             # Save name
           prune_ratio=0.5,           # Pruning Ratio (50%)
           prune_iterative_steps=1,   # Pruning Interative Steps
           sparse_training=False      # Experimental, Allow Sparse Training Before Pruning
)


# Quick Pruning (prune_epochs no need)
# prunetrain(quick_pruning=True, data='coco.yaml', train_epochs=10, imgsz=640, batch=8, device=[0], name='yolo11', 
#            prune_ratio=0.5, prune_iterative_steps=1)