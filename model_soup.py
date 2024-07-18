import torch


def model_soup(models, weights=None):
    """
    将多个模型通过平均它们的状态字典（state_dict）合并为一个模型。
    参数:
        models (list of torch.nn.Module): 待合并的模型列表。
        weights (list of float, optional): 如果需要加权平均，为每个模型指定权重。
    返回:
        torch.nn.Module: 合并后的单个模型，拥有平均后的权重。
    """
    # 初始化合并模型的状态字典为空字典
    soup_state_dict = {}

    # 如果未提供权重，使用等权重
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    # 遍历每个模型的状态字典及其对应的权重
    for model, weight in zip(models, weights):
        model_state_dict = model.state_dict()

        for key, value in model_state_dict.items():
            if key in soup_state_dict:
                soup_state_dict[key] += weight * value  # 累加权重的参数值
            else:
                soup_state_dict[key] = weight * value.clone()  # 克隆新的参数值

    # 根据模型列表中的一个模型创建一个新的模型实例，确保所有模型具有相同的架构
    soup_model = models[0].__class__()  # 创建一个新的模型实例
    soup_model.load_state_dict(soup_state_dict)  # 加载合并后的状态字典

    return soup_model
