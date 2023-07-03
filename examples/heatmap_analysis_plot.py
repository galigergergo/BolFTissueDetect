from binbagnets.plots import heatmap_plots


if __name__ == "__main__":
    img_path = 'datasets/EXAMPLE/original/test/img/387709-12692-44296.png'
    model_path = 'datasets/EXAMPLE/binary/LYM_aug/models/best_model.pth.tar'
    heatmap_plots.init_dataset('LUAD')
    heatmap_plots.plot_heatmap_analysis(img_path, 'binbagnet17', model_path, 2)
