import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

# Initial shape vertices
shape_vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]])

# Create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(right=0.75)

# Plot initial shape as a solid shape
shape = ax.fill(shape_vertices[:, 0], shape_vertices[:, 1], 'b')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title('Transformations and Matrices')

# Transformation matrix
T = np.eye(3)

# Function to update shape based on transformation matrix
def update_shape():
    global T
    transformed_vertices = np.dot(np.hstack([shape_vertices, np.ones((len(shape_vertices), 1))]), T.T)
    shape[0].set_xy(transformed_vertices[:, :2])
    fig.canvas.draw_idle()

# Transformation matrix text labels
ax_label_matrix = plt.axes([0.8, 0.15, 0.15, 0.15], frameon=False)
ax_label_matrix.get_xaxis().set_visible(False)
ax_label_matrix.get_yaxis().set_visible(False)
label_matrix = ax_label_matrix.text(0.5, 0.5, '', va='center', ha='center', fontsize=10)

# Function to update transformation matrix text labels
def update_matrix_label():
    if radio_buttons.value_selected == 'Translate':
        matrix_str = '[{:.3f}, {:.3f}, {:.3f}]\n[{:.3f}, {:.3f}, {:.3f}]\n[{:.3f}, {:.3f}, {:.3f}]'.format(
            T[0, 0], T[0, 1], T[0, 2],
            T[1, 0], T[1, 1], T[1, 2],
            T[2, 0], T[2, 1], T[2, 2]
    )
    else:
        matrix_str = '[{:.3f}, {:.3f}]\n[{:.3f}, {:.3f}]'.format(T[0, 0], T[0, 1], T[1, 0], T[1, 1])
    
    label_matrix.set_text(matrix_str)

# Slider update function
def update_slider(val):
    global T
    tx = slider_x.val
    ty = slider_y.val
    rotation = np.radians(slider_rotation.val)
    scale_x = slider_scale_x.val
    scale_y = slider_scale_y.val
    shear_x = slider_shear_x.val
    shear_y = slider_shear_y.val
    
    # Construct transformation matrix
    T = np.array([
        [scale_x, shear_x, tx],
        [shear_y, scale_y, ty],
        [0, 0, 1]
    ])
    T[:2, :2] = np.dot([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]], T[:2, :2])
    
    update_shape()
    update_matrix_label()

# Create sliders for transformation parameters
ax_slider_x = plt.axes([0.8, 0.65, 0.15, 0.03])
slider_x = Slider(ax_slider_x, 'X', -2.0, 2.0, valinit=0)
slider_x.on_changed(update_slider)

ax_slider_y = plt.axes([0.8, 0.6, 0.15, 0.03])
slider_y = Slider(ax_slider_y, 'Y', -2.0, 2.0, valinit=0)
slider_y.on_changed(update_slider)

ax_slider_rotation = plt.axes([0.8, 0.55, 0.15, 0.03])
slider_rotation = Slider(ax_slider_rotation, 'R', -180.0, 180.0, valinit=0)
slider_rotation.on_changed(update_slider)

ax_slider_scale_x = plt.axes([0.8, 0.5, 0.15, 0.03])
slider_scale_x = Slider(ax_slider_scale_x, 'X', 0.1, 2.0, valinit=1)
slider_scale_x.on_changed(update_slider)

ax_slider_scale_y = plt.axes([0.8, 0.45, 0.15, 0.03])
slider_scale_y = Slider(ax_slider_scale_y, 'Y', 0.1, 2.0, valinit=1)
slider_scale_y.on_changed(update_slider)

ax_slider_shear_x = plt.axes([0.8, 0.4, 0.15, 0.03])
slider_shear_x = Slider(ax_slider_shear_x, 'X', -1.0, 1.0, valinit=0)
slider_shear_x.on_changed(update_slider)

ax_slider_shear_y = plt.axes([0.8, 0.35, 0.15, 0.03])
slider_shear_y = Slider(ax_slider_shear_y, 'Y', -1.0, 1.0, valinit=0)
slider_shear_y.on_changed(update_slider)

# Function to handle radio button selection
def select_transformation(label):
    if label == 'Translate':
        slider_x.ax.set_visible(True)
        slider_y.ax.set_visible(True)
        slider_rotation.ax.set_visible(False)
        slider_scale_x.ax.set_visible(False)
        slider_scale_y.ax.set_visible(False)
        slider_shear_x.ax.set_visible(False)
        slider_shear_y.ax.set_visible(False)
    elif label == 'Rotate':
        slider_x.ax.set_visible(False)
        slider_y.ax.set_visible(False)
        slider_rotation.ax.set_visible(True)
        slider_scale_x.ax.set_visible(False)
        slider_scale_y.ax.set_visible(False)
        slider_shear_x.ax.set_visible(False)
        slider_shear_y.ax.set_visible(False)
    elif label == 'Scale':
        slider_x.ax.set_visible(False)
        slider_y.ax.set_visible(False)
        slider_rotation.ax.set_visible(False)
        slider_scale_x.ax.set_visible(True)
        slider_scale_y.ax.set_visible(True)
        slider_shear_x.ax.set_visible(False)
        slider_shear_y.ax.set_visible(False)
    elif label == 'Shear':
        slider_x.ax.set_visible(False)
        slider_y.ax.set_visible(False)
        slider_rotation.ax.set_visible(False)
        slider_scale_x.ax.set_visible(False)
        slider_scale_y.ax.set_visible(False)
        slider_shear_x.ax.set_visible(True)
        slider_shear_y.ax.set_visible(True)
    
    update_slider(None)
    update_matrix_label()

# Attach slider update function to each slider
slider_x.on_changed(update_slider)
slider_y.on_changed(update_slider)
slider_rotation.on_changed(update_slider)
slider_scale_x.on_changed(update_slider)
slider_scale_y.on_changed(update_slider)
slider_shear_x.on_changed(update_slider)
slider_shear_y.on_changed(update_slider)

# Create radio buttons for selecting transformations
ax_radio_buttons = plt.axes([0.8, 0.7, 0.15, 0.2])
radio_buttons = RadioButtons(ax_radio_buttons, ('Translate', 'Rotate', 'Scale', 'Shear'))
radio_buttons.on_clicked(select_transformation)

ax_reset_button = plt.axes([0.8, 0.05, 0.15, 0.05])
reset_button = Button(ax_reset_button, 'Reset')
def reset_sliders(event):
    slider_x.reset()
    slider_y.reset()
    slider_rotation.reset()
    slider_scale_x.reset()
    slider_scale_y.reset()
    slider_shear_x.reset()
    slider_shear_y.reset()
    update_slider(None)
reset_button.on_clicked(reset_sliders)

select_transformation('Translate')

plt.show()