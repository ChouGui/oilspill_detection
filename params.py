# PARAMETERS :
name = "mean1"
comment = "  "
xlabel = "batch size"
epochs = 50
bs = 16
lr = 0.0001
nbrruns = 5
models = ['resnet50','resnet101','vgg19','vgg16','densenet121']

# Dense layers
actis = ['relu','tanh']
d1s = [0, 2048]
d2s = [0, 512]
d3s = [0, 128]
# Data augmentation
shears = [0.,0.1,0.2,0.3,0.4,0.5]
zooms = [0.,0.1,0.2,0.3,0.4,0.5]
hflips = [True, False]
vflips = [False, True]
# Fitting
bss = [8,16,32,64,128]
lrs = [.001,.0005,.0001,.00005,.00001]

# WHAT TO TRY
xs = bss

# mail
mail = 'guillaume.legat@gmail.com'
passwd = 'wsocfhobvniljwyo'
