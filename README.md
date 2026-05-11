# FSIF-PCNN-full-spectrum-SIF-reconstruction-algorithm
Full-spectrum solar-induced chlorophyll fluorescence (SIF; 650–800 nm) provides a powerful means to characterize vegetation functional status. However, current retrieval methods still exhibit substantial uncertainties due to their reliance on fixed spectral-shape assumptions or strong priors for SIF and reflectance reconstruction, introducing inherent model-form biases when these assumptions mismatch observations. To address this limitation, we propose a physics-constrained neural network, FSIF-PCNN, for flexible and robust full-spectrum SIF retrieval. FSIF-PCNN leverages a U-Net architecture to extract multi-scale spectral features, while enforcing physical plausibility through radiative-transfer consistency constraints and spectral smoothness regularization. Therefore, embedding physics within a flexible deep-learning framework—thereby avoiding biases introduced by empirical spectral templates—enables more accurate, robust, and generalizable full-spectrum SIF retrieval. 

FSIF-PCNN is a physics-constrained neural network designed for full-spectrum solar-induced chlorophyll fluorescence (SIF) reconstruction over the 650–800 nm spectral range. The model is designed to reduce the dependence of conventional full-spectrum SIF retrieval methods on fixed spectral-shape assumptions or empirical SIF/reflectance priors. The model adopts a U-Net-based architecture to extract multi-scale spectral features from input radiance spectra, while enforcing physical consistency. FSIF-PCNN aims to provide a more flexible and robust framework for full-spectrum SIF retrieval by reducing dependence on fixed empirical spectral templates and improving the separation of fluorescence emission from reflected radiance.

# Key features
- Full-spectrum SIF reconstruction over 650–800 nm
- U-Net-based spectral feature extraction
- Physics-constrained learning framework
- Reduced reliance on fixed empirical spectral templates
