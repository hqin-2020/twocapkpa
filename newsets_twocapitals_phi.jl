#=============================================================================#
#  Economy with TWO CAPITAL STOCKS
#
#  Author: Balint Szoke
#  Date: Sep 2018
#=============================================================================#

using Pkg
using Optim
using Roots
using NPZ
using Distributed
using CSV
using Tables
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--gamma"
            help = "gamma"
            arg_type = Float64
            default = 8.0
        "--rho"
            help = "rho"
            arg_type = Float64
            default = 1.00001   
        "--kappa"
            help = "kappa"
            arg_type = Float64
            default = 0.0
        "--zeta"
            help = "zeta"
            arg_type = Float64
            default = 0.5  
        "--fraction"
            help = "fraction"
            arg_type = Float64
            default = 0.01   
        "--Delta"
            help = "Delta"
            arg_type = Float64
            default = 1000.  
        "--symmetric"
            help = "symmetric"
            arg_type = Int
            default = 0
        "--dataname"
            help = "dataname"
            arg_type = String
            default = "output"
        "--llim"
            help = "llim"
            arg_type = Float64
            default = 1.0
        "--lscale"
            help = "lscale"
            arg_type = Float64
            default = 1.0
        "--zscale"
            help = "zscale"
            arg_type = Float64
            default = 1.0
        "--foc"
            help = "foc"
            arg_type = Int
            default = 1
        "--clowerlim"
            help = "clowerlim"
            arg_type = Float64
            default = 0.0001
    end
    return parse_args(s)
end

#==============================================================================#
# SPECIFICATION:
#==============================================================================#
@show parsed_args = parse_commandline()
gamma                = parsed_args["gamma"]
rho                  = parsed_args["rho"]
kappa                = parsed_args["kappa"]
zeta                 = parsed_args["zeta"]
fraction             = parsed_args["fraction"]
Delta                = parsed_args["Delta"]
symmetric            = parsed_args["symmetric"]
dataname             = parsed_args["dataname"]
llim                 = parsed_args["llim"]
lscale                = parsed_args["lscale"]
zscale                = parsed_args["zscale"]
foc                   = parsed_args["foc"]
clowerlim             = parsed_args["clowerlim"]

symmetric_returns    = symmetric
state_dependent_xi   = 0
optimize_over_ell    = 0
compute_irfs         = 0                    # need to start julia with "-p 5"

if compute_irfs == 1
    @everywhere include("newsets_utils_phi.jl")
elseif compute_irfs ==0
    include("newsets_utils_phi.jl")
end

println("=============================================================")
if symmetric_returns == 1
    println(" Economy with two capital stocks: SYMMETRIC RETURNS          ")
    if state_dependent_xi == 0
        println(" No tilting (xi is NOT state dependent)                      ")
        filename = (compute_irfs==0) ? "model_sym_HS.npz" : "model_sym_HS_p.npz";
    elseif state_dependent_xi == 1
        println(" With tilting (change in kappa)                        ")
        filename = (compute_irfs==0) ? "model_sym_HSHS.npz" : "model_sym_HSHS_p.npz";
    elseif state_dependent_xi == 2
        println(" With tilting (change in beta)                        ")
        filename = (compute_irfs==0) ? "model_sym_HSHS2.npz" : "model_sym_HSHS2_p.npz";
    end
elseif symmetric_returns == 0
    println(" Economy with two capital stocks: ASYMMETRIC RETURNS         ")
    if state_dependent_xi == 0
        println(" No tilting (xi is NOT state dependent)                      ")
        filename = (compute_irfs==0) ? "model_asym_HS.npz" : "model_asym_HS_p.npz";
    elseif state_dependent_xi == 1
        println(" With tilting (change in kappa)                        ")
        filename = (compute_irfs==0) ? "model_asym_HSHS.npz" : "model_asym_HSHS_p.npz";
    elseif state_dependent_xi == 2
        println(" With tilting (change in beta)                        ")
        filename = (compute_irfs==0) ? "model_asym_HSHS2.npz" : "model_asym_HSHS2_p.npz";
    end
end

filename_ell = "./output/"*dataname*"/Delta_"*string(Delta)*"_llim_"*string(llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(kappa)*"_zeta_"*string(zeta)*"/gamma_"*string(gamma)*"_rho_"*string(rho)*"/"
isdir(filename_ell) || mkpath(filename_ell)

#==============================================================================#
#  PARAMETERS
#==============================================================================#

# (1) Baseline model
a11 = 0.014
alpha = 0.1

scale = 1.32
sigma_k1 = scale*[.0048,               .0,   .0];
sigma_k2 = scale*[.0              , .0048,   .0];

scale = sqrt(1.754)
# scale = sqrt(1.0)
sigma_k1 = scale*[.00477,               .0,   .0];
sigma_k2 = scale*[.0              , .00477,   .0];
sigma_z =  [.011*sqrt(.5)   , .011*sqrt(.5)   , .025];

eta1 = 0.013
eta2 = 0.013
eta1 = 0.012790328319261378
eta2 = 0.012790328319261378
if symmetric_returns == 1
    beta1 = 0.01
    beta2 = 0.01
    # beta1 = 0.005
    # beta2 = 0.005
else
    beta1 = 0.0
    beta2 = 0.01
    # beta1 = 0.0
    # beta2 = 0.005
end

delta = .002;
# delta = .0025;

phi1 = 28.0
phi2 = 28.0

# (3) GRID
II, JJ = trunc(Int,1000*lscale+1), trunc(Int,200*zscale+1);
rmax = llim;#log(20);
rmin = -llim;#-log(20); 
# rmax = 3.;#log(20);
# rmin = -3.;#-log(20); 
zmax = 1.;
# zmax = 0.5;
zmin = -zmax;

# (4) Iteration parameters
maxit = 50000;        # maximum number of iterations in the HJB loop
# maxit = 6;        # maximum number of iterations in the HJB loop
crit  = 10e-6;      # criterion HJB loop
# crit  = 10e-3;      # criterion HJB loop
# Delta = 1000.;      # delta in HJB algorithm


# Initialize model objects -----------------------------------------------------
baseline1 = Baseline(a11, zeta, kappa, sigma_z, beta1, eta1, sigma_k1, delta);
baseline2 = Baseline(a11, zeta, kappa, sigma_z, beta2, eta2, sigma_k2, delta);
technology1 = Technology(alpha, phi1);
technology2 = Technology(alpha, phi2);
model = TwoCapitalEconomy(baseline1, baseline2, technology1, technology2);

grid = Grid_rz(rmin, rmax, II, zmin, zmax, JJ);
params = FinDiffMethod(maxit, crit, Delta);

#==============================================================================#
# WITH ROBUSTNESS
#==============================================================================#

preload = 0

if preload == 1
    preload_kappa = kappa
    preload_llim = 18.0#1.0#llim
    preload_rho = 1.0
    # if symmetric_returns == 1
    #     preload_Delta = 300.0
    # else
    #     preload_Delta = 100.0
    # end
    preload_Delta = 300.0
    preload_gamma = gamma
    preload_zeta = 0.5
    # preloadname = "./output/"*"twocap_re_calib_kappa_0"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(preload_gamma)*"_rho_"*string(preload_rho)*"/"
    # preloadname = "./output/"*"twocap_re_calib_re_init"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(preload_gamma)*"_rho_"*string(preload_rho)*"/"
    # preloadname = "./output/"*"twocap_re_calib"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(preload_gamma)*"_rho_"*string(preload_rho)*"/"
    # preloadname = "./output/"*"twocap_special_1_c_foc"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(preload_gamma)*"_rho_"*string(preload_rho)*"/"
    # preload = npzread(preloadname*filename)
    preloadname = "/project/lhansen/twocapbal/output/Standard_grid_sym_Delta_300_scale_1754_tol_21_15_scale_3_300_frac_0.0/gamma_"*string(preload_gamma)*"_rho_1.0/"
    preload = npzread(preloadname*"model_sym_HS.npz")
    println("preload location : "*preloadname)
    preloadV0 = preload["V"]
    preloadd1 = preload["d1"]
    preloadcons = preload["cons"]
else
    if rho == 1.0
        preloadV0 = -2*ones(grid.I, grid.J)
        preloadd1 = 0.03*ones(grid.I, grid.J)
        preloadcons = 0.03*ones(grid.I, grid.J)
    else
        preload_kappa = kappa
        preload_rho = rho# 1.0#1.0#1.00001#0.8#0.95#0.9#
        preload_Delta = Delta#100.0
        preload_zeta = 0.5
        preload_llim = llim#1.0
        # preloadname = "./output/"*"twocap_c_foc_clowerlim_03_c_init"*"/Delta_"*string(preload_Delta)*"_llim_"*string(llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        preloadname = "./output/"*"twocap_re_calib_kappa_0"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        # preloadname = "./output/"*"twocap_re_calib"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        # preloadname = "./output/"*"twocap_c_foc_clowerlim_03_c_init_18_15"*"/Delta_"*string(preload_Delta)*"_llim_"*string(llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        preload = npzread(preloadname*filename)
        println("preload location : "*preloadname)
        preloadV0 = preload["V"]
        preloadd1 = preload["d1"]
        preloadcons = preload["cons"]
    end
end

println(" (3) Compute value function WITH ROBUSTNESS")
times = @elapsed begin
A, V, val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, h1_B, h2_B, hz_B,
        mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z, V0, Vr, Vr_F, Vr_B, Vz_B, Vz_F, cF, cB, Vz, rr, zz, pii, dr, dz =
        value_function_twocapitals(gamma, rho, fraction, model, grid, params, preloadV0, preloadd1, preloadcons, foc, clowerlim, symmetric_returns);
println("=============================================================")
end
println("Convegence time (minutes): ", times/60)
g_dist, g = stationary_distribution(A, grid)

# Define Policies object
policies  = PolicyFunctions(d1_F, d2_F, d1_B, d2_B,
                            -h1_F, -h2_F, -hz_F,
                            -h1_B, -h2_B, -hz_B);

# Construct drift terms under the baseline
mu_1 = (mu_1_F + mu_1_B)/2.;
mu_r = (mu_r_F + mu_r_B)/2.;
# ... under the worst-case model
h1_dist = (policies.h1_F + policies.h1_B)/2.;
h2_dist = (policies.h2_F + policies.h2_B)/2.;
hz_dist = (policies.hz_F + policies.hz_B)/2.;

######
d1 = (policies.d1_F + policies.d1_B)/2;
d2 = (policies.d2_F + policies.d2_B)/2;
h1, h2, hz = -h1_dist, -h2_dist, -hz_dist;


r = range(rmin, stop=rmax, length=II);    # capital ratio vector
rr = r * ones(1, JJ);
pii = rr;
IJ = II*JJ;
k1a = zeros(II,JJ)
k2a = zeros(II,JJ)
for i=1:IJ
    p = pii[i];
    k1a[i] = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
    k2a[i] = ((1-zeta)*exp.(p*((kappa-1))) + zeta).^(1/(kappa-1));
end
d1k = d1.*k1a
d2k = d2.*k2a
c = alpha*ones(II,JJ) - d1k - d2k
# CSV.write(filename_ell*"d1.csv",  Tables.table(d1), writeheader=false)
# CSV.write(filename_ell*"d2.csv",  Tables.table(d2), writeheader=false)
# CSV.write(filename_ell*"h1.csv",  Tables.table(h1), writeheader=false)
# CSV.write(filename_ell*"h2.csv",  Tables.table(h2), writeheader=false)
# CSV.write(filename_ell*"hz.csv",  Tables.table(hz), writeheader=false)

results = Dict("delta" => delta,
# Two capital stocks
"eta1" => eta1, "eta2" => eta2, "a11"=> a11, 
"beta1" => beta1, "beta2" => beta2,
"k1a" => k1a, "k2a"=> k2a,
"d1k" => d1k, "d2k"=> d2k,
"sigma_k1" => sigma_k1, "sigma_k2" => sigma_k2,
"sigma_z" =>  sigma_z, "alpha" => alpha, "kappa" => kappa,"zeta" => zeta, "phi1" => phi1, "phi2" => phi2,
"I" => II, "J" => JJ,
"rmax" => rmax, "rmin" => rmin, "zmax" => zmax, "zmin" => zmin,
"rr" => rr, "zz" => zz, "pii" => pii, "dr" => dr, "dz" => dz,
"maxit" => maxit, "crit" => crit, "Delta" => Delta,
"g_dist" => g_dist, "g" => g,
"cons" => c,
# Robust control under baseline
"V0" => V0, "V" => V, "Vr" => Vr, "Vr_F" => Vr_F, "Vr_B" => Vr_B, "Vz" => Vz, "Vz_F" => Vz_F, "Vz_B" => Vz_B, "val" => val, "gamma" => gamma, "rho" => rho,
"d1" => d1, "d2" => d2, "d1_F" => d1_F, "d2_F" => d2_F, "d1_B" => d1_B, "d2_B" => d2_B, "cF" => cF, "cB" => cB,
"h1_F" => h1_F, "h2_F" => h2_F, "hz_F" => hz_F, "h1_B" => h1_B, "h2_B" => h2_B, "hz_B" => hz_B, 
"mu_1_F" => mu_1_F, "mu_1_B" => mu_1_B, "mu_r_F" => mu_r_F, "mu_r_B" => mu_r_B, 
"h1" => h1, "h2" => h2, "hz" => hz, "foc" => foc, "clowerlim" => clowerlim,  "zscale" => zscale, "lscale" => lscale, "llim" => llim,
"times" => times,
"mu_1" => mu_1, "mu_r" => mu_r, "mu_z" => mu_z)

npzwrite(filename_ell*filename, results)
