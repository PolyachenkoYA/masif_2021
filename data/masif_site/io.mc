##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Thu Aug 20 20:36:08 2020
##############################################################################
Hello world from PE 0
Vnm_tstart: starting timer 26 (APBS WALL CLOCK)..
NOsh_parseInput:  Starting file parsing...
NOsh: Parsing READ section
NOsh: Storing molecule 0 path 2UUY-uR-50_A
NOsh: Done parsing READ section
NOsh: Done parsing READ section (nmol=1, ndiel=0, nkappa=0, ncharge=0, npot=0)
NOsh: Parsing ELEC section
NOsh_parseMG: Parsing parameters for MG calculation
NOsh_parseMG:  Parsing dime...
PBEparm_parseToken:  trying dime...
MGparm_parseToken:  trying dime...
NOsh_parseMG:  Parsing cglen...
PBEparm_parseToken:  trying cglen...
MGparm_parseToken:  trying cglen...
NOsh_parseMG:  Parsing fglen...
PBEparm_parseToken:  trying fglen...
MGparm_parseToken:  trying fglen...
NOsh_parseMG:  Parsing cgcent...
PBEparm_parseToken:  trying cgcent...
MGparm_parseToken:  trying cgcent...
NOsh_parseMG:  Parsing fgcent...
PBEparm_parseToken:  trying fgcent...
MGparm_parseToken:  trying fgcent...
NOsh_parseMG:  Parsing mol...
PBEparm_parseToken:  trying mol...
NOsh_parseMG:  Parsing lpbe...
PBEparm_parseToken:  trying lpbe...
NOsh: parsed lpbe
NOsh_parseMG:  Parsing bcfl...
PBEparm_parseToken:  trying bcfl...
NOsh_parseMG:  Parsing pdie...
PBEparm_parseToken:  trying pdie...
NOsh_parseMG:  Parsing sdie...
PBEparm_parseToken:  trying sdie...
NOsh_parseMG:  Parsing srfm...
PBEparm_parseToken:  trying srfm...
NOsh_parseMG:  Parsing chgm...
PBEparm_parseToken:  trying chgm...
MGparm_parseToken:  trying chgm...
NOsh_parseMG:  Parsing sdens...
PBEparm_parseToken:  trying sdens...
NOsh_parseMG:  Parsing srad...
PBEparm_parseToken:  trying srad...
NOsh_parseMG:  Parsing swin...
PBEparm_parseToken:  trying swin...
NOsh_parseMG:  Parsing temp...
PBEparm_parseToken:  trying temp...
NOsh_parseMG:  Parsing calcenergy...
PBEparm_parseToken:  trying calcenergy...
NOsh_parseMG:  Parsing calcforce...
PBEparm_parseToken:  trying calcforce...
NOsh_parseMG:  Parsing write...
PBEparm_parseToken:  trying write...
NOsh_parseMG:  Parsing end...
MGparm_check:  checking MGparm object of type 1.
NOsh:  nlev = 4, dime = (161, 129, 129)
NOsh: Done parsing ELEC section (nelec = 1)
NOsh: Parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing file (got QUIT)
Valist_readPQR: Counted 3220 atoms
Valist_getStatistics:  Max atom coordinate:  (63.53, 59.12, 57.1)
Valist_getStatistics:  Min atom coordinate:  (12.87, 15.91, 22.01)
Valist_getStatistics:  Molecule center:  (38.2, 37.515, 39.555)
NOsh_setupCalcMGAUTO(nosh.c, 1589):  coarse grid center = 38.2 37.515 39.555
NOsh_setupCalcMGAUTO(nosh.c, 1594):  fine grid center = 38.2 37.515 39.555
NOsh_setupCalcMGAUTO (nosh.c, 1606):  Coarse grid spacing = 0.561106, 0.607617, 0.491539
NOsh_setupCalcMGAUTO (nosh.c, 1608):  Fine grid spacing = 0.455063, 0.513672, 0.445391
NOsh_setupCalcMGAUTO (nosh.c, 1610):  Displacement between fine and coarse grids = 0, 0, 0
NOsh:  2 levels of focusing with 0.81101, 0.845387, 0.906114 reductions
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalcMGAUTO (nosh.c, 1704):  starting mesh repositioning.
NOsh_setupCalcMGAUTO (nosh.c, 1706):  coarse mesh center = 38.2 37.515 39.555
NOsh_setupCalcMGAUTO (nosh.c, 1711):  coarse mesh upper corner = 83.0885 76.4025 71.0135
NOsh_setupCalcMGAUTO (nosh.c, 1716):  coarse mesh lower corner = -6.6885 -1.3725 8.0965
NOsh_setupCalcMGAUTO (nosh.c, 1721):  initial fine mesh upper corner = 74.605 70.39 68.06
NOsh_setupCalcMGAUTO (nosh.c, 1726):  initial fine mesh lower corner = 1.795 4.64 11.05
NOsh_setupCalcMGAUTO (nosh.c, 1787):  final fine mesh upper corner = 74.605 70.39 68.06
NOsh_setupCalcMGAUTO (nosh.c, 1792):  final fine mesh lower corner = 1.795 4.64 11.05
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalc:  Mapping ELEC statement 0 (1) to calculation 1 (2)
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 29.4633
Vpbe_ctor2:  solute dimensions = 52.81 x 45.75 x 37.01
Vpbe_ctor2:  solute charge = 6
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 74 table
Vclist_ctor2:  Using 75 x 75 x 74 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (61.736, 54.286, 46.166)
Vclist_setupGrid:  Grid lower corner = (7.332, 10.372, 16.472)
Vclist_assignAtoms:  Have 3673554 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 191.134
Vacc_storeParms:  Using 1936-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_ctor2:  PMG chose nx = 161, ny = 129, nz = 129
Vpmg_ctor2:  PMG chose nlev = 4
Vpmg_ctor2:  PMG chose nxc = 21, nyc = 17, nzc = 17
Vpmg_ctor2:  PMG chose nf = 2679201, nc = 6069
Vpmg_ctor2:  PMG chose narr = 3072144, narrc = 392943
Vpmg_ctor2:  PMG chose n_rpc = 500, n_iz = 250, n_ipc = 500
Vpmg_ctor2:  PMG chose nrwk = 34280405, niwk = 750
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vpmg_fillco:  done filling coefficient arrays
Vpmg_fillco:  filling boundary arrays
Vpmg_fillco:  done filling boundary arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 1.267909e+00
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (MGDRIV2: fine pro)..
Vnm_tstop: stopping timer 30 (MGDRIV2: fine pro).  CPU TIME = 8.526800e-02
Vnm_tstart: starting timer 30 (MGDRIV2: coarse pro)..
Vnm_tstop: stopping timer 30 (MGDRIV2: coarse pro).  CPU TIME = 8.367990e-01
Vnm_tstart: starting timer 30 (MGDRIV2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.248749e+00
PMG: iteration =  0
PMG: relative residual =  1.000000e+00
PMG: contraction number =  1.000000e+00
PMG: iteration =  1
PMG: relative residual =  1.196730e-01
PMG: contraction number =  1.196730e-01
PMG: iteration =  2
PMG: relative residual =  1.575639e-02
PMG: contraction number =  1.316620e-01
PMG: iteration =  3
PMG: relative residual =  2.290772e-03
PMG: contraction number =  1.453869e-01
PMG: iteration =  4
PMG: relative residual =  3.662355e-04
PMG: contraction number =  1.598742e-01
PMG: iteration =  5
PMG: relative residual =  7.037280e-05
PMG: contraction number =  1.921518e-01
PMG: iteration =  6
PMG: relative residual =  1.833787e-05
PMG: contraction number =  2.605818e-01
PMG: iteration =  7
PMG: relative residual =  5.278420e-06
PMG: contraction number =  2.878426e-01
PMG: iteration =  8
PMG: relative residual =  2.002812e-06
PMG: contraction number =  3.794341e-01
PMG: iteration =  9
PMG: relative residual =  7.002407e-07
PMG: contraction number =  3.496287e-01
Vnm_tstop: stopping timer 30 (MGDRIV2: solve).  CPU TIME = 1.250388e+01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 1.346164e+01
Vpmg_setPart:  lower corner = (-6.6885, -1.3725, 8.0965)
Vpmg_setPart:  upper corner = (83.0885, 76.4025, 71.0135)
Vpmg_setPart:  actual minima = (-6.6885, -1.3725, 8.0965)
Vpmg_setPart:  actual maxima = (83.0885, 76.4025, 71.0135)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 1.021270068710E+05 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 4.207000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 29.4633
Vpbe_ctor2:  solute dimensions = 52.81 x 45.75 x 37.01
Vpbe_ctor2:  solute charge = 6
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 74 table
Vclist_ctor2:  Using 75 x 75 x 74 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (61.736, 54.286, 46.166)
Vclist_setupGrid:  Grid lower corner = (7.332, 10.372, 16.472)
Vclist_assignAtoms:  Have 3673554 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 191.134
Vacc_storeParms:  Using 1936-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_ctor2:  PMG chose nx = 161, ny = 129, nz = 129
Vpmg_ctor2:  PMG chose nlev = 4
Vpmg_ctor2:  PMG chose nxc = 21, nyc = 17, nzc = 17
Vpmg_ctor2:  PMG chose nf = 2679201, nc = 6069
Vpmg_ctor2:  PMG chose narr = 3072144, narrc = 392943
Vpmg_ctor2:  PMG chose n_rpc = 500, n_iz = 250, n_ipc = 500
Vpmg_ctor2:  PMG chose nrwk = 34280405, niwk = 750
Vpmg_ctor2:  Filling boundary with old solution!
VPMG::focusFillBound -- New mesh mins = 1.795, 4.64, 11.05
VPMG::focusFillBound -- New mesh maxs = 74.605, 70.39, 68.06
VPMG::focusFillBound -- Old mesh mins = -6.6885, -1.3725, 8.0965
VPMG::focusFillBound -- Old mesh maxs = 83.0885, 76.4025, 71.0135
VPMG::extEnergy:  energy flag = 1
Vpmg_setPart:  lower corner = (1.795, 4.64, 11.05)
Vpmg_setPart:  upper corner = (74.605, 70.39, 68.06)
Vpmg_setPart:  actual minima = (-6.6885, -1.3725, 8.0965)
Vpmg_setPart:  actual maxima = (83.0885, 76.4025, 71.0135)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
VPMG::extEnergy:   Finding extEnergy dimensions...
VPMG::extEnergy    Disj part lower corner = (1.795, 4.64, 11.05)
VPMG::extEnergy    Disj part upper corner = (74.605, 70.39, 68.06)
VPMG::extEnergy    Old lower corner = (-6.6885, -1.3725, 8.0965)
VPMG::extEnergy    Old upper corner = (83.0885, 76.4025, 71.0135)
Vpmg_qmEnergy:  Zero energy for zero ionic strength!
VPMG::extEnergy: extQmEnergy = 0 kT
Vpmg_qfEnergyVolume:  Calculating energy
VPMG::extEnergy: extQfEnergy = 0 kT
VPMG::extEnergy: extDiEnergy = 0.692479 kT
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vpmg_fillco:  done filling coefficient arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 1.271271e+00
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (MGDRIV2: fine pro)..
Vnm_tstop: stopping timer 30 (MGDRIV2: fine pro).  CPU TIME = 7.014800e-02
Vnm_tstart: starting timer 30 (MGDRIV2: coarse pro)..
Vnm_tstop: stopping timer 30 (MGDRIV2: coarse pro).  CPU TIME = 7.062650e-01
Vnm_tstart: starting timer 30 (MGDRIV2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.696143e+01
PMG: iteration =  0
PMG: relative residual =  1.000000e+00
PMG: contraction number =  1.000000e+00
PMG: iteration =  1
PMG: relative residual =  1.325274e-01
PMG: contraction number =  1.325274e-01
PMG: iteration =  2
PMG: relative residual =  1.725917e-02
PMG: contraction number =  1.302310e-01
PMG: iteration =  3
PMG: relative residual =  2.492813e-03
PMG: contraction number =  1.444341e-01
PMG: iteration =  4
PMG: relative residual =  3.831910e-04
PMG: contraction number =  1.537183e-01
PMG: iteration =  5
PMG: relative residual =  6.866345e-05
PMG: contraction number =  1.791886e-01
PMG: iteration =  6
PMG: relative residual =  1.716888e-05
PMG: contraction number =  2.500440e-01
PMG: iteration =  7
PMG: relative residual =  4.918328e-06
PMG: contraction number =  2.864676e-01
PMG: iteration =  8
PMG: relative residual =  1.850592e-06
PMG: contraction number =  3.762645e-01
PMG: iteration =  9
PMG: relative residual =  6.574287e-07
PMG: contraction number =  3.552532e-01
Vnm_tstop: stopping timer 30 (MGDRIV2: solve).  CPU TIME = 1.244907e+01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 1.329721e+01
Vpmg_setPart:  lower corner = (1.795, 4.64, 11.05)
Vpmg_setPart:  upper corner = (74.605, 70.39, 68.06)
Vpmg_setPart:  actual minima = (1.795, 4.64, 11.05)
Vpmg_setPart:  actual maxima = (74.605, 70.39, 68.06)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 1.340823848715E+05 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 8.297000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vgrid_writeDX:  Opening virtual socket...
Vgrid_writeDX:  Writing to virtual socket...
Vgrid_writeDX:  Writing comments for ASC format.
printEnergy:  Performing global reduction (sum)
Vcom_reduce:  Not compiled with MPI, doing simple copy.
Vnm_tstop: stopping timer 26 (APBS WALL CLOCK).  CPU TIME = 3.041089e+01
##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Thu Aug 20 20:36:15 2020
##############################################################################
Vgrid_readDX:  Grid dimensions 161 x 129 x 129 grid
Vgrid_readDX:  Grid origin = (1.795, 4.64, 11.05)
Vgrid_readDX:  Grid spacings = (0.455062, 0.513672, 0.445391)
Vgrid_readDX:  allocating 161 x 129 x 129 doubles for storage
