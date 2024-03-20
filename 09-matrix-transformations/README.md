Matrix Transformations for 2D Shapes
This project implements matrix transformations of translation, rotation, scaling, and shearing on a 2D shape along the X and Y axes. Each transformation requires a specific size of the transformation matrix:

    For translation of a 2D shape, a 3x3 transformation matrix is required.
    For other transformations (rotation, scaling, and shearing), a 2x2 matrix is sufficient.

By adjusting the sliders corresponding to each transformation, the elements in the transformation matrix change accordingly. These changes are observable in the shape's transformation, demonstrating how altering one or more elements in the matrix facilitates different types of transformations.

For 3D shapes, an additional dimension in the matrix would be necessary to accommodate the extra axis of transformation.

Installation:
To run this tool, ensure you have numpy and matplotlib installed. You can install it using pip:
pip install numpy
pip install matplotlib