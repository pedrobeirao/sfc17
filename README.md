## Overview

The SFC17 model is a tool to model SFC (Specific Fuel Consumption) curves using a combination of robust linear regression and gaussian processes. It allows you to set many parameters: engine numbers, power range and step size, and allows separate modeling of different fuel types. 

The SFC17 model consists of two stages:

- Robust linear model

- Non-linear, non-parametric model that fits residuals from the purely linear model.

The data is cleaned and rebinned to 5 min intervals, and then one or 2 SFC models are fitted, depending on wehter we have 1 or 2 fuel types in the data.

The SFC17 package is structured in 3 folders: 
- "Models": contains the implementation of the core regression and GP modeling functions
- "Tools": contains cleaning and rebinning, fuel type split, engine info, json conversion, etc.
- "Validation": contains validation plot functions.

You can interact with the SFC17 package using the Python notebook under the sfc17 folder.
