# Given:
$bearing\_std = 0.35$ (default_value) \
initial robot command $u=[\delta_{rot1}, \delta_{trans}, \delta_{rot2}]^T=[0,10,0]^T$ \
initial mean state $\mu_1=[x,y,\theta]^T=[180,50,0]^T$ \
alphas $\sqrt \alpha = [0.05, 0.001, 0.05, 0.01]^T$ squared root of alphas, used for transition noise in action space ($M_t$)

# Calculate
## Covariance $Q$
Given only bearing and signature (id of the landmark) \
 $Q = \begin{pmatrix} \delta_{\phi}^2 & 0 \\ 0 & \delta_{s}^2 \end{pmatrix} = \begin{pmatrix}  0.35^2 & 0 \\ 0 & 0 \end{pmatrix}$ - noise covariance (corresponds to landmark definition)

## Covariance $R_t$. 

### Transition function:
$x_t = g(x_{t-1},u_t,\epsilon_t=0)=\left[\begin{matrix}x_{t-1}+\delta_{trans}cos(\theta+\delta_{rot1}) \\ y_{t-1}+\delta_{trans}sin(\theta+\delta_{rot1}) \\ \theta_{t-1}+\delta_{rot1}+\delta_{rot2} \end{matrix}\right]$

### Covariance of noise added to action space:
$\epsilon_t \thicksim N(0,M_t)$ \
$M_t=\left[\begin{matrix}\epsilon_{\delta_{rot1}} \\ \epsilon_{\delta_{trans}} \\ \epsilon_{\delta_{rot2}} \end{matrix}\right]=\left[\begin{matrix}\alpha_1\delta_{rot1}^2+\alpha_2\delta_{trans}^2 & 0 & 0 \\ 0 & \alpha_3\delta_{trans}^2+\alpha_4(\delta_{rot1}^2+\delta_{rot2}^2) & 0 \\ 0 & 0 & \alpha_1\delta_{rot2}^2+\alpha_2\delta_{trans}^2\end{matrix}\right]=\left[\begin{matrix}0.05^2*0^2+0.001^2*10^2 & 0 & 0 \\ 0 & 0.05^2*10^2+0.01^2*(0^2+0^2) & 0 \\ 0 & 0 & 0.05^2*0^2+0.001^2*10^2 \end{matrix}\right]=\left[\begin{matrix} 0.0001 & 0 & 0 \\ 0 & 0.25 & 0 \\ 0 & 0 & 0.0001 \end{matrix}\right]$ 

### Jacobians $G_t$ and $V_t$ with respect to $x_{t-1}$ and $u_t$ respectively
$G_t=\frac{\partial g(x_{t-1},u_t,\epsilon_t)}{\partial x_{t-1}}|_{\mu_{t-1},\epsilon_t=0}=\left[\begin{matrix}1 & 0 & -\delta_{trans}sin(\theta+\delta_{rot1}) \\ 0 & 1 & \delta_{trans}cos(\theta+\delta_{rot1}) \\ 0 & 0 & 1 \end{matrix}\right]=\left[\begin{matrix}1 & 0 & -10*sin(0+0) \\ 0 & 1 & 10*cos(0+0) \\ 0 & 0 & 1 \end{matrix}\right]=\left[\begin{matrix}1 & 0 & 0 \\ 0 & 1 & 10 \\ 0 & 0 & 1 \end{matrix}\right]$ 

$V_t=\frac{\partial g(x_{t-1},u_t,\epsilon_t)}{\partial u_t}|_{\mu_{t-1},\epsilon_t=0}=\left[\begin{matrix} -\delta_{tras}sin(\theta+\delta_{rot1}) & cos(\theta+\delta_{rot1}) & 0 \\ \delta_{trans}cos(\theta+\delta_{rot1}) & sin(\theta+\delta_{rot1}) & 0 \\ 1 & 0 & 1 \end{matrix}\right]=\left[\begin{matrix}-10*sin(0+0) & cos(0+0) & 0 \\ 10*cos(0+0) & sin(0+0) & 0 \\ 1 & 0 & 1 \end{matrix}\right]=\left[\begin{matrix} 0 & 1 & 0 \\ 10 & 0 & 0 \\ 1 & 0 & 1 \end{matrix}\right]$


### Jacobian $H_t$ with respect to $x_t$ corresponds to the observation function
$H_t=\frac{\partial h(x_t)}{x_t}|_{\overline \mu_t=g(\mu_{t-1},u_t)}=\left[\begin{matrix} \frac{-(m_{j,x}-\overline \mu_{t,x})}{\sqrt{(m_{j,x}-\overline \mu_{t,x})^2+(m_{j,y}-\overline \mu_{t,y})^2}} & \frac{-(m_{j,y}-\overline \mu_{t,y})}{\sqrt{(m_{j,x}-\overline \mu_{t,x})^2+(m_{j,y}-\overline \mu_{t,y})^2}} & 0 \\ \frac{m_{j,y}-\overline \mu_{t,y}}{(m_{j,x}-\overline \mu_{t,x})^2+(m_{j,y}-\overline \mu_{t,y})^2} & \frac{-(m_{j,x}-\overline \mu_{t,x})}{(m_{j,x}-\overline \mu_{t,x})^2+(m_{j,y}-\overline \mu_{t,y})^2} & -1 \end{matrix}\right]$

### Finally,
$R_t = V@M_t@V^T=\left[\begin{matrix} 0 & 1 & 0 \\ 10 & 0 & 0 \\ 1 & 0 & 1 \end{matrix}\right]@\left[\begin{matrix} 0.0001 & 0 & 0 \\ 0 & 0.25 & 0 \\ 0 & 0 & 0.0001 \end{matrix}\right]@\left[\begin{matrix} 0 & 10 & 1 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{matrix}\right]=\left[\begin{matrix} 0.25 & 0 & 0 \\ 0 & 0.01 & 0.001 \\ 0 & 0.01 & 0.0002 \end{matrix}\right]$

