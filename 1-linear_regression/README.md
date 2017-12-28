# Linear Regression

## Variables
  - **&theta;** - minimizing parameters to be found
  - **X** - input varibles [Mx(N+1)]
    - x<sub>i0</sub> = 1 para i = 1, 2, ... N
  - **y** - lables (nx1)

## Hypothesis

![f1]

## Cost function

![f2]

## Gradient Descent

![f3]

[f1]: http://chart.apis.google.com/chart?cht=tx&chl=\mathbf{h}=\mathbf{h_\theta(X)}=\mathbf{X\theta}

[f2]: http://chart.apis.google.com/chart?cht=tx&chl=minJ(\mathbf{\theta})=\frac{1}{2m}(\mathbf{h-y})^2

[f3]: http://chart.apis.google.com/chart?cht=tx&chl=\mathbf{\theta}=\mathbf{\theta}-\frac{\alpha}{m}\mathbf{X}^T(\mathbf{h-y})
