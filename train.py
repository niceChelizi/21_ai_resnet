net=resnet50(class_num=cfg.num_class)
#计算softmax交叉熵。
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
#设置Adam优化器
train_step_size = de_train.get_dataset_size()
lr = Tensor(get_lr(global_step=0, total_epochs=cfg.epoch_size, steps_per_epoch=train_step_size))
opt = Momentum(net.trainable_params(), lr, momentum=0.9, weight_decay=1e-4, loss_scale=cfg.loss_scale_num)
loss_scale = FixedLossScaleManager(cfg.loss_scale_num, False)
model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})
loss_cb = LossMonitor(per_print_times=train_step_size)
ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps, keep_checkpoint_max=1)
ckpoint_cb = ModelCheckpoint(prefix=cfg.prefix, directory=cfg.directory, config=ckpt_config)
print("train:")
model.train(100, de_train, callbacks=[loss_cb,ckpoint_cb], dataset_sink_mode=True)