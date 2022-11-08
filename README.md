# Linear Combination of Interpolating Wavelets
Please note the following points before using the porgram.

## Note:

1) Provide a file containing the points of a surface $f(x,y)$ in the following format, "x y f(x,y)", 

$-0.31415900E+01\quad -0.31415900E+01\quad 0.44635719E+02$\
$-0.31415900E+01\quad -0.30781235E+01\quad  0.28100063E+01$\
$-0.31415900E+01\quad -0.30146571E+01\quad  0.62991668E+01$\
$-0.31415900E+01\quad -0.29511906E+01\quad  0.71413888E+01$\
$-0.31415900E+01\quad -0.28877241E+01\quad  0.81301097E+01$
    
2) Currently there are no provisions to change the paramters of the model through user interface but will be implemented soon.

3) This program reconstructs the surface by fitting the surface derivatives with the derivatives of the interpolating wavelets. This may not be a straight forward method of reconstruction but may come handy at certain situations. 
