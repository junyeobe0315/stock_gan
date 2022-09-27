import torch
import matplotlib.pyplot as plt

from train import *

best_predictor = 13500
checkpoint = torch.load(os.path.join("./models_fc_disc/","model_epoch_{}.pt".format(best_predictor)))
generator.load_state_dict(checkpoint['generator_model_state_dict'])
rmse_values = model_rmse(generator, test_dataloader, epoch=best_predictor, plot_graph=True, plot_title="Test Predictions", show_preds=True)
print(rmse_values)
