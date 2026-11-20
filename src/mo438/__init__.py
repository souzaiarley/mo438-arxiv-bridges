"""MO438 (Unicamp) — structural analysis of the ogbn-arxiv citation network.

One package for the whole project:

* data and graphs: :mod:`data` (OGB download/parsing), :mod:`graph_build` (G_d -> G_u, the
  single source of truth of the conversion), :mod:`graph_io` (GraphML writer) and
  :mod:`instances` (strict GraphML reader used by the analyses);
* objective 1, structural characterization: :mod:`characterization` (pipeline),
  :mod:`metrics`, :mod:`paths`, :mod:`clustering`, and the independent re-check
  :mod:`verification` (numpy/scipy only, no igraph and no :mod:`metrics` code);
* objectives 2-7: :mod:`edges`, :mod:`flows`, :mod:`overlap`, :mod:`removal`,
  :mod:`removal_experiment`, :mod:`lcc_distances`, :mod:`communities`, :mod:`ordering`,
  :mod:`integration` and :mod:`review`;
* shared: :mod:`config`, :mod:`runtime` (logging, provenance, JSON), :mod:`plots`,
  :mod:`reference` (values published in the F1 report) and :mod:`cli` (``mo438`` command).
"""

__version__ = "0.2.0"
