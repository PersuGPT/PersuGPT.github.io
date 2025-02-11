LOCALES = {
    "lang": {
        "zh": {
            "label": "Lang"
        },
        "zh_bk": {
            "label": "语言"
        }
    },
    "model_name": {
        "zh": {
            "label": "Model name"
        },
        "zh_bk": {
            "label": "模型名称"
        }
    },
    "model_path": {
        "zh": {
            "label": "Model path",
            "info": "Path to pretrained model or model identifier from Hugging Face."
        },
        "zh_bk": {
            "label": "模型路径",
            "info": "本地模型的文件路径或 Hugging Face 的模型标识符。"
        }
    },
    "finetuning_type": {
        "zh": {
            "label": "Finetuning method"
        },
        "zh_bk": {
            "label": "微调方法"
        }
    },
    "adapter_path": {
        "zh": {
            "label": "Adapter path"
        },
        "zh_bk": {
            "label": "适配器路径"
        }
    },
    "refresh_btn": {
        "zh": {
            "value": "Refresh adapters"
        },
        "zh_bk": {
            "value": "刷新适配器"
        }
    },
    "advanced_tab": {
        "zh": {
            "label": "Advanced configurations"
        },
        "zh_bk": {
            "label": "高级设置"
        }
    },
    "quantization_bit": {
        "zh": {
            "label": "Quantization bit",
            "info": "Enable 4/8-bit model quantization (QLoRA)."
        },
        "zh_bk": {
            "label": "量化等级",
            "info": "启用 4/8 比特模型量化（QLoRA）。"
        }
    },
    "template": {
        "zh": {
            "label": "Prompt template",
            "info": "The template used in constructing prompts."
        },
        "zh_bk": {
            "label": "提示模板",
            "info": "构建提示词时使用的模板"
        }
    },
    "rope_scaling": {
        "zh": {
            "label": "RoPE scaling"
        },
        "zh_bk": {
            "label": "RoPE 插值方法"
        }
    },
    "booster": {
        "zh": {
            "label": "Booster"
        },
        "zh_bk": {
            "label": "加速方式"
        }
    },
    "training_stage": {
        "zh": {
            "label": "Stage",
            "info": "The stage to perform in training."
        },
        "zh_bk": {
            "label": "训练阶段",
            "info": "目前采用的训练方式。"
        }
    },
    "dataset_dir": {
        "zh": {
            "label": "Data dir",
            "info": "Path to the data directory."
        },
        "zh_bk": {
            "label": "数据路径",
            "info": "数据文件夹的路径。"
        }
    },
    "dataset": {
        "zh": {
            "label": "Dataset"
        },
        "zh_bk": {
            "label": "数据集"
        }
    },
    "data_preview_btn": {
        "zh": {
            "value": "Preview dataset"
        },
        "zh_bk": {
            "value": "预览数据集"
        }
    },
    "preview_count": {
        "zh": {
            "label": "Count"
        },
        "zh_bk": {
            "label": "数量"
        }
    },
    "page_index": {
        "zh": {
            "label": "Page"
        },
        "zh_bk": {
            "label": "页数"
        }
    },
    "prev_btn": {
        "zh": {
            "value": "Prev"
        },
        "zh_bk": {
            "value": "上一页"
        }
    },
    "next_btn": {
        "zh": {
            "value": "Next"
        },
        "zh_bk": {
            "value": "下一页"
        }
    },
    "close_btn": {
        "zh": {
            "value": "Close"
        },
        "zh_bk": {
            "value": "关闭"
        }
    },
    "preview_samples": {
        "zh": {
            "label": "Samples"
        },
        "zh_bk": {
            "label": "样例"
        }
    },
    "cutoff_len": {
        "zh": {
            "label": "Cutoff length",
            "info": "Max tokens in input sequence."
        },
        "zh_bk": {
            "label": "截断长度",
            "info": "输入序列分词后的最大长度。"
        }
    },
    "learning_rate": {
        "zh": {
            "label": "Learning rate",
            "info": "Initial learning rate for AdamW."
        },
        "zh_bk": {
            "label": "学习率",
            "info": "AdamW 优化器的初始学习率。"
        }
    },
    "num_train_epochs": {
        "zh": {
            "label": "Epochs",
            "info": "Total number of training epochs to perform."
        },
        "zh_bk": {
            "label": "训练轮数",
            "info": "需要执行的训练总轮数。"
        }
    },
    "max_samples": {
        "zh": {
            "label": "Max samples",
            "info": "Maximum samples per dataset."
        },
        "zh_bk": {
            "label": "最大样本数",
            "info": "每个数据集最多使用的样本数。"
        }
    },
    "compute_type": {
        "zh": {
            "label": "Compute type",
            "info": "Whether to use fp16 or bf16 mixed precision training."
        },
        "zh_bk": {
            "label": "计算类型",
            "info": "是否启用 FP16 或 BF16 混合精度训练。"
        }
    },
    "batch_size": {
        "zh": {
            "label": "Batch size",
            "info": "Number of samples to process per GPU."
        },
        "zh_bk":{
            "label": "批处理大小",
            "info": "每块 GPU 上处理的样本数量。"
        }
    },
    "gradient_accumulation_steps": {
        "zh": {
            "label": "Gradient accumulation",
            "info": "Number of gradient accumulation steps."
        },
        "zh_bk": {
            "label": "梯度累积",
            "info": "梯度累积的步数。"
        }
    },
    "lr_scheduler_type": {
        "zh": {
            "label": "LR Scheduler",
            "info": "Name of learning rate scheduler.",
        },
        "zh_bk": {
            "label": "学习率调节器",
            "info": "采用的学习率调节器名称。"
        }
    },
    "max_grad_norm": {
        "zh": {
            "label": "Maximum gradient norm",
            "info": "Norm for gradient clipping.."
        },
        "zh_bk": {
            "label": "最大梯度范数",
            "info": "用于梯度裁剪的范数。"
        }
    },
    "val_size": {
        "zh": {
            "label": "Val size",
            "info": "Proportion of data in the dev set."
        },
        "zh_bk": {
            "label": "验证集比例",
            "info": "验证集占全部样本的百分比。"
        }
    },
    "extra_tab": {
        "zh": {
            "label": "Extra configurations"
        },
        "zh_bk": {
            "label": "其它参数设置"
        }
    },
    "logging_steps": {
        "zh": {
            "label": "Logging steps",
            "info": "Number of steps between two logs."
        },
        "zh_bk": {
            "label": "日志间隔",
            "info": "每两次日志输出间的更新步数。"
        }
    },
    "save_steps": {
        "zh": {
            "label": "Save steps",
            "info": "Number of steps between two checkpoints."
        },
        "zh_bk": {
            "label": "保存间隔",
            "info": "每两次断点保存间的更新步数。"
        }
    },
    "warmup_steps": {
        "zh": {
            "label": "Warmup steps",
            "info": "Number of steps used for warmup."
        },
        "zh_bk": {
            "label": "预热步数",
            "info": "学习率预热采用的步数。"
        }
    },
    "neftune_alpha": {
        "zh": {
            "label": "NEFTune Alpha",
            "info": "Magnitude of noise adding to embedding vectors."
        },
        "zh_bk": {
            "label": "NEFTune 噪声参数",
            "info": "嵌入向量所添加的噪声大小。"
        }
    },
    "train_on_prompt": {
        "zh": {
            "label": "Train on prompt",
            "info": "Compute loss on the prompt tokens in supervised fine-tuning."
        },
        "zh_bk": {
            "label": "计算输入损失",
            "info": "在监督微调时候计算输入序列的损失。"
        }
    },
    "upcast_layernorm": {
        "zh": {
            "label": "Upcast LayerNorm",
            "info": "Upcast weights of layernorm in float32."
        },
        "zh_bk": {
            "label": "缩放归一化层",
            "info": "将归一化层权重缩放至 32 位浮点数。"
        }
    },
    "lora_tab": {
        "zh": {
            "label": "LoRA configurations"
        },
        "zh_bk": {
            "label": "LoRA 参数设置"
        }
    },
    "lora_rank": {
        "zh": {
            "label": "LoRA rank",
            "info": "The rank of LoRA matrices."
        },
        "zh_bk": {
            "label": "LoRA 秩",
            "info": "LoRA 矩阵的秩。"
        }
    },
    "lora_dropout": {
        "zh": {
            "label": "LoRA Dropout",
            "info": "Dropout ratio of LoRA weights."
        },
        "zh_bk": {
            "label": "LoRA 随机丢弃",
            "info": "LoRA 权重随机丢弃的概率。"
        }
    },
    "lora_target": {
        "zh": {
            "label": "LoRA modules (optional)",
            "info": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules."
        },
        "zh_bk": {
            "label": "LoRA 作用模块（非必填）",
            "info": "应用 LoRA 的目标模块名称。使用英文逗号分隔多个名称。"
        }
    },
    "additional_target": {
        "zh": {
            "label": "Additional modules (optional)",
            "info": "Name(s) of modules apart from LoRA layers to be set as trainable. Use commas to separate multiple modules."
        },
        "zh_bk": {
            "label": "附加模块（非必填）",
            "info": "除 LoRA 层以外的可训练模块名称。使用英文逗号分隔多个名称。"
        }
    },
    "create_new_adapter": {
        "zh": {
            "label": "Create new adapter",
            "info": "Whether to create a new adapter with randomly initialized weight or not."
        },
        "zh_bk": {
            "label": "新建适配器",
            "info": "是否创建一个经过随机初始化的新适配器。"
        }
    },
    "rlhf_tab": {
        "zh": {
            "label": "RLHF configurations"
        },
        "zh_bk": {
            "label": "RLHF 参数设置"
        }
    },
    "dpo_beta": {
        "zh": {
            "label": "DPO beta",
            "info": "Value of the beta parameter in the DPO loss."
        },
        "zh_bk": {
            "label": "DPO beta 参数",
            "info": "DPO 损失函数中 beta 超参数大小。"
        }
    },
    "reward_model": {
        "zh": {
            "label": "Reward model",
            "info": "Adapter of the reward model for PPO training. (Needs to refresh adapters)"
        },
        "zh_bk": {
            "label": "奖励模型",
            "info": "PPO 训练中奖励模型的适配器路径。（需要刷新适配器）"
        }
    },
    "cmd_preview_btn": {
        "zh": {
            "value": "Preview command"
        },
        "zh_bk": {
            "value": "预览命令"
        }
    },
    "start_btn": {
        "zh": {
            "value": "Start"
        },
        "zh_bk": {
            "value": "开始"
        }
    },
    "rand_btn": {
        "zh": {
            "value": "Random Selected Tasks"
        },
        "zh_bk": {
            "value": "随机选择说服任务"
        }
    },
    "stop_btn": {
        "zh": {
            "value": "Abort"
        },
        "zh_bk": {
            "value": "中断"
        }
    },
    "output_dir": {
        "zh": {
            "label": "Output dir",
            "info": "Directory for saving results."
        },
        "zh_bk": {
            "label": "输出目录",
            "info": "保存结果的路径。"
        }
    },
    "output_box": {
        "zh": {
            "value": "Ready."
        },
        "zh_bk": {
            "value": "准备就绪。"
        }
    },
    "loss_viewer": {
        "zh": {
            "label": "Loss"
        },
        "zh_bk": {
            "label": "损失"
        }
    },
    "predict": {
        "zh": {
            "label": "Save predictions"
        },
        "zh_bk": {
            "label": "保存预测结果"
        }
    },
    "load_btn": {
        "zh": {
            "value": "Load model"
        },
        "zh_bk": {
            "value": "加载模型"
        }
    },
    "unload_btn": {
        "zh": {
            "value": "Unload model"
        },
        "zh_bk": {
            "value": "卸载模型"
        }
    },
    "info_box": {
        "zh": {
            "value": "Model unloaded, please load a model first."
        },
        "zh_bk": {
            "value": "模型未加载，请先加载模型。"
        }
    },
    "system": {
        "zh": {
            "placeholder": "System prompt (optional)"
        },
        "zh_bk": {
            "placeholder": "系统提示词（非必填）"
        }
    },
    "persuader": {
        "zh": {
            "label": "Persuader",
            "placeholder": "Required"
        },
        "zh_bk": {
            "label": "说服者",
            "placeholder": "必填，示例：银行营销女生"
        }
    },
    "context": {
        "zh": {
            "label": "Persuasion Scenario",
            "placeholder": "Required, e.g., You are a volunteer for the charity SaveTheChild. Now a programmer is hesitating whether to donate."
        },
        "zh_bk": {
            "label": "说服背景",
            "placeholder": "必填，示例：你是银行的营销女生，一位退休老太太来银行存钱，你要向她推销一款理财产品，但老太太坚持定期存款"
        }
    },
    "Task": {
        "zh": {
            "label": "Persuasion Goal",
            "placeholder": "Required, e.g., Persuade programmers to donate"
        },
        "zh_bk": {
            "label": "说服目标",
            "placeholder": "必填，示例：说服退休老太太购买理财产品"
        }
    },
    "query": {
        "zh": {
            "placeholder": "Input..."
        },
        "zh_bk": {
            "placeholder": "输入..."
        }
    },
    "submit_btn": {
        "zh": {
            "value": "Submit"
        },
        "zh_bk": {
            "value": "发送"
        }
    },
    "clear_btn": {
        "zh": {
            "value": "Clear History"
        },
        "zh_bk": {
            "value": "清空历史"
        }
    },
    "max_length": {
        "zh": {
            "label": "Maximum length"
        },
        "zh_bk": {
            "label": "最大长度"
        }
    },
    "max_new_tokens": {
        "zh": {
            "label": "Maximum new tokens"
        },
        "zh_bk": {
            "label": "最大生成长度"
        }
    },
    "top_p": {
        "zh": {
            "label": "Top-p"
        },
        "zh_bk": {
            "label": "Top-p 采样值"
        }
    },
    "temperature": {
        "zh": {
            "label": "Temperature"
        },
        "zh_bk": {
            "label": "温度系数"
        }
    },
     "repetition_penalty": {
        "zh": {
            "label": "Repetition penalty"
        },
        "zh_bk": {
            "label": "重复惩罚参数"
        }
    },
    "max_shard_size": {
        "zh": {
            "label": "Max shard size (GB)",
            "info": "The maximum size for a model file."
        },
        "zh_bk": {
            "label": "最大分块大小（GB）",
            "info": "单个模型文件的最大大小。"
        }
    },
    "export_quantization_bit": {
        "zh": {
            "label": "Export quantization bit.",
            "info": "Quantizing the exported model."
        },
        "zh_bk": {
            "label": "导出量化等级",
            "info": "量化导出模型。"
        }
    },
    "export_quantization_dataset": {
        "zh": {
            "label": "Export quantization dataset.",
            "info": "The calibration dataset used for quantization."
        },
        "zh_bk": {
            "label": "导出量化数据集",
            "info": "量化过程中使用的校准数据集。"
        }
    },
    "export_dir": {
        "zh": {
            "label": "Export dir",
            "info": "Directory to save exported model."
        },
        "zh_bk": {
            "label": "导出目录",
            "info": "保存导出模型的文件夹路径。"
        }
    },
    "export_btn": {
        "zh": {
            "value": "Export"
        },
        "zh_bk": {
            "value": "开始导出"
        }
    }
}


ALERTS = {
    "err_conflict": {
        "zh": "A process is in running, please abort it firstly.",
        "zh_bk": "任务已存在，请先中断训练。"
    },
    "err_exists": {
        "zh": "You have loaded a model, please unload it first.",
        "zh_bk": "模型已存在，请先卸载模型。"
    },
    "err_no_model": {
        "zh": "Please select a model.",
        "zh_bk": "请选择模型。"
    },
    "err_no_path": {
        "zh": "Model not found.",
        "zh_bk": "模型未找到。"
    },
    "err_no_dataset": {
        "zh": "Please choose a dataset.",
        "zh_bk": "请选择数据集。"
    },
    "err_no_adapter": {
        "zh": "Please select an adapter.",
        "zh_bk": "请选择一个适配器。"
    },
    "err_no_export_dir": {
        "zh": "Please provide export dir.",
        "zh_bk": "请填写导出目录"
    },
    "err_failed": {
        "zh": "Failed.",
        "zh_bk": "训练出错。"
    },
    "err_demo": {
        "zh": "Training is unavailable in demo mode, duplicate the space to a private one first.",
        "zh_bk": "展示模式不支持训练，请先复制到私人空间。"
    },
    "info_aborting": {
        "zh": "Aborted, wait for terminating...",
        "zh_bk": "训练中断，正在等待线程结束……"
    },
    "info_aborted": {
        "zh": "Ready.",
        "zh_bk": "准备就绪。"
    },
    "info_finished": {
        "zh": "Finished.",
        "zh_bk": "训练完毕。"
    },
    "info_loading": {
        "zh": "Loading model...",
        "zh_bk": "加载中……"
    },
    "info_unloading": {
        "zh": "Unloading model...",
        "zh_bk": "卸载中……"
    },
    "info_loaded": {
        "zh": "Model loaded, now you can chat with your model!",
        "zh_bk": "模型已加载，可以开始聊天了！"
    },
    "info_unloaded": {
        "zh": "Model unloaded.",
        "zh_bk": "模型已卸载。"
    },
    "info_exporting": {
        "zh": "Exporting model...",
        "zh_bk": "正在导出模型……"
    },
    "info_exported": {
        "zh": "Model exported.",
        "zh_bk": "模型导出完成。"
    }
}
