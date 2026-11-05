# M3B figures, reference-library style v2

These figures are a presentation-only redraw of the frozen M3B boundary analysis.
The numerical source tables, candidate set, labels, and model predictions are unchanged.

- `M3B_threshold_response_v2.*`: six-condition faceted response plot. Black circles are the fixed grid; magenta diamonds are model-guided enriched points. The pale green band marks the actual BSM1 compliance region (`SNH_max_eval <= 4 mg N/L`). Lines join frozen model predictions as visual guides only.
- `M3B_fixed_grid_performance_v2.*`: fixed-grid RMSE, raw threshold errors, Gate reference fraction, and near-threshold RMSE.
- `boundary_*_plot_data_v2.csv`: exact data used for the plotted marks and bars.
- `make_boundary_figures_v2.py`: reproducible renderer. It rechecks the M3B freeze and source/output hashes before drawing.

Formats are PNG (320 dpi), TIFF (600 dpi, LZW), editable SVG, and PDF.
