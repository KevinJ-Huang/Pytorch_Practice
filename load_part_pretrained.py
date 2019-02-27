    pretrained_dict = torch.load(os.path.join(config.SAVE,config.TASK,config.NAME,'snapshots','pretrained.pth'))
    model_dict = config.model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    config.model.load_state_dict(model_dict)
    print("load model.....", os.path.join(config.SAVE, config.TASK, config.NAME, 'snapshots', 'pretrained.pth'))
