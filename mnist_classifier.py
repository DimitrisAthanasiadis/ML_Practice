from fastai.vision import *

# if __name__ == '__main__':
bs = 64  # bs = 16 # αν μου τελειωσει η μνημη

path = untar_data(URLs.MNIST_SAMPLE)  # εκει που βρισκεται το συνολο δεδομενων
path.ls()
(path / 'train').ls()

# df = pd.read_csv(path/'labels.csv') # οι ετικετες των δεδομενων
# path_img = df['name']

tfms = get_transforms(do_flip=False)  # transforms-φτιαχνει καπως τις εικονες ετσι ωστε να ειναι επεξεργασιμες
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)  # data
data.normalize()
data.show_batch(rows=3, figsize=(5, 5))

# εκπαιδευση
learn = cnn_learner(data, models.resnet50, metrics=error_rate)  # ο μαθητης
# learn.model # η εκπαιδευση
learn.freeze()
learn.fit(1)  # πιο αποδοτικο το fit_one_cycle, 4 εποχες
learn.unfreeze()
learn.save("stage-1")
learn.show_results()# σωζει το σταδιο που ειμαστε με ονομα stage-1 αν θελουμε να κανουμε φορτωση ξανα απο εδω

# αποτελεσματα
interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
interp.plt_top_losses(2, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
interp.most_confused(min_val=2)

# learn.unfreeze() # ξεκλειδωνουμε το dataset για παραπανω δεδομενα
# learn.fit_one_cycle(1)
# learn.load('stage-1')
# learn.lr_find()
# learn.recorder.plot()
# learn.unfreeze()
# learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))
