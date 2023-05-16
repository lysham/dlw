%% Data readme and sample matlab code
% for Shakespeare and Roderick, submitted to J. Climate, 2021

%% Fig 4 - effective optical depth as a function of q and H_eff
% tau_eff_[CO2](q,Heff)
% tau_eff_[CO2] : 2d array : no units; [CO2]=200,400,600,800ppm provided
% q : 1d array : kg/kg
% Heff : 1d array : m

%% Fig 5g - time mean scale height of water vapour from ERA5
% H(lon,lat)
% H : 2d array : m
% lon : 1d array : -180 to 180 degrees
% lat : 1d array : degrees

%% Reference values
P0=1e5; % kPa
sigma = 5.67e-8; % W/m^2/K^4

%% Calculation method: given T1,q1,P1, and H1 or position (lon1,lat1)

% If H1 is unknown, interpolate from ERA5 H(lon,lat)
H1=interp2(lon,lat',H',lon1,lat1);

% effective height (Eqn. 11)
Heff1=H1./cos(40.3*pi/180).*(P1/P0).^1.8;

% interpolate to find tau1 at (q1,Heff1) 
tau1=interp2(q,Heff',tau_eff_[CO2],q1,Heff1);

% calculate  longwave (Eqn. 8)
L=(1-exp(-tau1))*sigma*T1.^4; % W/m^2






