def supervised_loss(data, attrs, selected_attrs):
    pos = {}
    mapping = {}
    loss = {}
    loss_negative = {}
    total_loss = 0.0
    i = 0
    for attr in selected_attrs:
        pos[attr] = data[data['attr'] == attr]['ix'].to_numpy()
        mapping[i] = attr
        attr_pos = torch.index_select(attrs[i],0, torch.tensor(pos[attr]))
        pos_loss = (1 - cos_sim(attr_pos, attr_pos).flatten().to(torch.float16)).sum()
        loss[attr] = pos_loss.detach().numpy()
        total_loss += pos_loss

        attr_neg = torch.index_select(attrs[i],0, torch.tensor(pos_std))
        neg_loss = cos_sim(attr_pos, attr_neg).flatten().to(torch.float16).sum()
        loss_negative[attr] = neg_loss.detach().numpy()
        total_loss += neg_loss

    return total_loss, loss, mapping, loss_negative