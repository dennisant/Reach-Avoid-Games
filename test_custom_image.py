import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

def draw_real_car(path, x, y, theta):
    transform_data = Affine2D().rotate_deg_around(*(x0, y0), theta) + plt.gca().transData
    ax.imshow(
        plt.imread(path, format="png"), 
        transform = transform_data, 
        interpolation='none',
        origin='lower',
        extent=[x0 - 0.927, x0 + 3.34, y0 - 0.944, y0 + 1.044],
        alpha = 0.5,
        clip_on=True)

paths = ['visual_components/.png', 'visual_components/delorean.png', 'visual_components/delorean.png', 'visual_components/delorean.png']
thetas = [90, 60, 45, 30]
x = [0, 2, 4, 6]
y = [0, 2, 4, 6]
fig, ax = plt.subplots()
for x0, y0, path, theta in zip(x, y, paths, thetas):
    draw_real_car(path, x0, y0, theta)

plt.xticks(range(0, 20))
plt.yticks(range(0, 20))
plt.show()