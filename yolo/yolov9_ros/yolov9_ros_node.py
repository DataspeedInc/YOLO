#!/usr/bin/python3
import sys
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import gs_ui_msgs
from gs_ui_msgs.msg import BaseConfig, StartTraining
import threading
from rclpy.executors import MultiThreadedExecutor

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from lightning.pytorch.loggers import CSVLogger

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config, NMSConfig, DatasetConfig, ModelConfig, TrainConfig, InferenceConfig, ValidationConfig, DataConfig, OptimizerConfig, LossConfig, SchedulerConfig, EMAConfig, MatcherConfig
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.utils.logging_utils import setup


class YoloTrainingNode(Node):
    def __init__(self):
        super().__init__('yolov9_ros_node')
        
        # ROS 2 Parameter initialization (optional)
        self.trainer = None
        self.cfg = None
        self.train_cfg = None
        self.inference_cfg = None
        self.validation_cfg = None
        self.dataset_cfg = None
        self.model_cfg = None
        self.nms_cfg = None
        self.data_cfg = None
        self.optimizer_cfg = None
        self.loss_cfg = None
        self.scheduler_cfg = None
        self.ema_cfg = None
        self.epochs = 0
        self.cfg_path = project_root / 'yolo' / 'config'
        
        
        self.get_logger().info("Starting YOLO training node...")
        self.start_train_sub = self.create_subscription(StartTraining, 'yolov9/start_training', self.run_training, 1)
        self.metrics_pub = self.create_publisher(String, "yolov9/metrics", 10)

    def run_training(self, msg):
        # Setup config
        base_cfg = BaseConfig()
        
        model = OmegaConf.load(self.cfg_path / 'model' /  'v9-s.yaml')
        dataset = OmegaConf.load(self.cfg_path / 'dataset' / 'dev.yaml')
        # model_dict = OmegaConf.to_container(model, resolve=True)
        # self.model_cfg = ModelConfig(name = model_dict.get('name', {}), anchor=model_dict.get('anchor', {}), model=model_dict.get('model', {}))
        base_cfg = msg
        val_data_cfg = DataConfig(batch_size=1,image_size=[640, 640],cpu_num=16,shuffle=True,pin_memory=True,source='dev',data_augment={},dynamic_shape=False)
        self.nms_cfg = NMSConfig(min_confidence=0.25, min_iou=0.65, max_bbox=300)
        self.validation_cfg = ValidationConfig(task='validation', nms=self.nms_cfg, data=val_data_cfg)
        ema_config = EMAConfig(enable=True, decay=0.995)
        scheduler_config = SchedulerConfig(type='LinearLR', warmup={'epochs': 3}, args={'total_iters': 2000, 'start_factor': 1, 'end_factor': 0.01})
        matcher_config = MatcherConfig(iou='CIoU', topk=10, factor={'iou': 6.0, 'cls': 0.5})
        self.loss_cfg = LossConfig(objective={'BoxLoss': 7.5, 'DFLoss': 1.5, 'BCELoss': 0.5},aux=0.25,matcher=matcher_config)
        self.optimizer_cfg = OptimizerConfig(type='SGD', args={'lr': 0.001, 'weight_decay': 0.0005, 'momentum': 0.937, 'nesterov': True})
        self.dataset_cfg = DatasetConfig(path='data/flow', class_num=1, class_list=['flowers'], auto_download=None)
        self.data_cfg = DataConfig(batch_size=1,image_size=[640, 640],cpu_num=16,shuffle=True,pin_memory=True,data_augment={},dynamic_shape=False,source='dev')
        self.train_cfg = TrainConfig(task='train',epoch=2000,data=self.data_cfg,optimizer=self.optimizer_cfg,loss=self.loss_cfg,scheduler=scheduler_config,ema=ema_config,validation=self.validation_cfg)
        self.cfg = Config(name='flower-v9-s', task=self.train_cfg, dataset=dataset, model=model, device='cuda', image_size=[640, 640], out_path='runs', exist_ok=True, lucky_number=10, use_wandb=False, use_tensorboard=False, weight=True, cpu_num=16)
        
        
        callbacks, loggers, save_path = setup(self.cfg)
        csv_logger = CSVLogger(save_dir=save_path)
        loggers.append(csv_logger)
        
        early_stop = EarlyStopping(monitor="Loss/BoxLoss_epoch", mode="min", patience=50)
        callbacks.append(early_stop)
        model_checkpoint = ModelCheckpoint(dirpath=save_path, filename="best", monitor="PyCOCO/AP @ .5:.95", mode="max", save_last=True, auto_insert_metric_name=True)
        callbacks.append(model_checkpoint)
        # ros_metrics_publisher = ROS2MetricsPublisher()
        # callbacks.append(ros_metrics_publisher)
        lambda_cb = LambdaCallback(on_train_epoch_end=lambda trainer, pl_module: self.pub_metrics(trainer, pl_module))
        callbacks.append(lambda_cb)
        
        
        self.trainer = Trainer(
            accelerator="auto",
            max_epochs=self.cfg.task.epoch,
            precision="16-mixed",
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=1,
            gradient_clip_val=10,
            gradient_clip_algorithm="value",
            deterministic=True,
            enable_progress_bar=True,
            default_root_dir=save_path,
        )

        if self.cfg.task.task == "train":
            model = TrainModel(self.cfg)
            self.trainer.fit(model)
        elif self.cfg.task.task == "validation":
            model = ValidateModel(self.cfg)
            self.trainer.validate(model)
        elif self.cfg.task.task == "inference":
            model = InferenceModel(self.cfg)
            self.trainer.predict(model)
            
    def pub_metrics(self, trainer, pl_module):
        metrics_dict = trainer.logged_metrics
        metrics_dict['epoch'] = trainer.current_epoch
        metrics_string = ", ".join([f"{key}: {value}" for key, value in metrics_dict.items()])
        # self.get_logger().info(f"Epoch {trainer.current_epoch}: {metrics_string}")
        self.metrics_pub.publish(String(data=metrics_string))

def spin_node(executor, node):
    rclpy.spin(node)
    
def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS 2 system
    
    # Create the node
    yolo_node = YoloTrainingNode()
    
    rclpy.spin(yolo_node)
    
    # # Clean up and shutdown when done
    yolo_node.destroy_node()
    
    rclpy.shutdown()



if __name__ == "__main__":
    main()
