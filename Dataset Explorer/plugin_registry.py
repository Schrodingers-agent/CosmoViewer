"""
plugin_registry.py
──────────────────
Central registry of all known plugins.

To add a new quantity:
  1. Write MyPlugin(BasePlugin) in its own file.
  2. Import it here.
  3. Add one line to REGISTRY.
  4. Done — the tree, tabs, and renderer adapt automatically.

REGISTRY
--------
  key   : (GROUP, DATASET)  e.g. ("PartType0", "Density")
  value : plugin class (not instance)

build_tree_schema()
-------------------
  Merges TREE_CONTRIBUTIONS from all registered plugins into the
  fixed display order defined by FIXED_GROUP_ORDER.
  Returns the same structure FileTreePanel expects:
    list of (group_name, [leaf, …])
"""

from __future__ import annotations

from plugin_base import BasePlugin
from generic_plugin import SKIP_GROUPS, configure_units, make_generic_leaves
from density_plugin      import DensityPlugin
from metallicity_plugin  import MetallicityPlugin

# BH Plugins
from bh_cuml_accr_mass_plugin import BHCumlAccrMassPlugin
from bh_density_plugin        import BHDensityPlugin
from bh_mass_plugin           import BHMassPlugin, BHParticleMassPlugin
from bh_accretion_rate_plugin import BHAccretionRatePlugin

# Gas Physics Plugins
from internal_energy_plugin         import InternalEnergyPlugin
from gas_mass_plugin                import GasMassPlugin
from max_temperature_plugin         import MaxTemperaturePlugin
from entropy_plugin                 import EntropyPlugin
from scale_factor_max_temp_plugin   import ScaleFactorMaxTempPlugin

# Gas Element Abundance Plugins
from abundance_carbon_plugin    import AbundanceCarbonPlugin
from abundance_helium_plugin    import AbundanceHeliumPlugin
from abundance_hydrogen_plugin  import AbundanceHydrogenPlugin
from abundance_iron_plugin      import AbundanceIronPlugin
from abundance_magnesium_plugin import AbundanceMagnesiumPlugin
from abundance_neon_plugin      import AbundanceNeonPlugin
from abundance_nitrogen_plugin  import AbundanceNitrogenPlugin
from abundance_oxygen_plugin    import AbundanceOxygenPlugin
from abundance_silicon_plugin   import AbundanceSiliconPlugin

# Stellar Plugins
from stellar_birth_density_plugin   import StellarBirthDensityPlugin
from stellar_initial_mass_plugin    import StellarInitialMassPlugin
from particle_id_plugin             import ParticleIDPlugin
from scale_factor_at_birth_plugin   import ScaleFactorAtBirthPlugin

# Research-hub analyses (measurements/), shown under "Analyses"
from analysis_plugins               import ANALYSES
from analysis_plugins_m2            import ANALYSES_M2

REGISTRY: dict[tuple[str, str], type[BasePlugin]] = {
    # Original
    ("PartType0", "Density"):     DensityPlugin,
    ("PartType0", "Metallicity"): MetallicityPlugin,

    # Black Hole
    ("PartType5", "BH_CumlAccrMass"): BHCumlAccrMassPlugin,
    ("PartType5", "BH_Density"):      BHDensityPlugin,
    ("PartType5", "BH_Mass"):         BHMassPlugin,
    ("PartType5", "Mass"):            BHParticleMassPlugin,
    ("PartType5", "BH_Mdot"):         BHAccretionRatePlugin,

    # Gas Physics
    ("PartType0", "InternalEnergy"):         InternalEnergyPlugin,
    ("PartType0", "Mass"):                   GasMassPlugin,
    ("PartType0", "MaximumTemperature"):     MaxTemperaturePlugin,
    ("PartType0", "Entropy"):                EntropyPlugin,
    ("PartType0", "AExpMaximumTemperature"): ScaleFactorMaxTempPlugin,

    # Gas Element Abundances
    ("PartType0", "ElementAbundance/Carbon"):    AbundanceCarbonPlugin,
    ("PartType0", "ElementAbundance/Helium"):    AbundanceHeliumPlugin,
    ("PartType0", "ElementAbundance/Hydrogen"):  AbundanceHydrogenPlugin,
    ("PartType0", "ElementAbundance/Iron"):      AbundanceIronPlugin,
    ("PartType0", "ElementAbundance/Magnesium"): AbundanceMagnesiumPlugin,
    ("PartType0", "ElementAbundance/Neon"):      AbundanceNeonPlugin,
    ("PartType0", "ElementAbundance/Nitrogen"):  AbundanceNitrogenPlugin,
    ("PartType0", "ElementAbundance/Oxygen"):    AbundanceOxygenPlugin,
    ("PartType0", "ElementAbundance/Silicon"):   AbundanceSiliconPlugin,

    # Stellar
    ("PartType4", "BirthDensity"):         StellarBirthDensityPlugin,
    ("PartType4", "InitialMass"):          StellarInitialMassPlugin,
    ("PartType4", "ParticleIDs"):          ParticleIDPlugin,
    ("PartType4", "StellarFormationTime"): ScaleFactorAtBirthPlugin,

    # Analyses: computed from several datasets by measurements/
    **ANALYSES,
    **ANALYSES_M2,
}

# ── Fixed group display order ─────────────────────────────────────────────────
# Groups listed here always appear in this order in the tree.
# Groups found in the HDF5 file but NOT listed here appear at the bottom.
FIXED_GROUP_ORDER = [
    "Config",
    "Header",
    "Parameters",
    "PartType0",
    "PartType1",
    "PartType2",
    "PartType3",
    "PartType4",
    "PartType5",
    "Analyses",
]

# Groups that are always shown (even with no leaves) in the fixed order
ALWAYS_SHOW = {"Config", "Header", "Parameters",
               "PartType0", "PartType1", "PartType2",
               "PartType3", "PartType4", "PartType5", "Analyses"}


def build_tree_schema() -> list[tuple[str, list[str]]]:
    """
    Merge all plugin TREE_CONTRIBUTIONS into a fixed-order schema.

    Returns list of (group_name, [leaf_names_from_all_plugins]).
    Leaf order within each group follows REGISTRY insertion order.
    """
    contributions: dict[str, list[str]] = {}
    for plugin_cls in REGISTRY.values():
        for group, leaves in plugin_cls.TREE_CONTRIBUTIONS.items():
            existing = contributions.setdefault(group, [])
            for leaf in leaves:
                if leaf not in existing:
                    existing.append(leaf)

    schema: list[tuple[str, list[str]]] = []
    seen:   set[str]                    = set()

    for group in FIXED_GROUP_ORDER:
        schema.append((group, contributions.get(group, [])))
        seen.add(group)

    for group, leaves in contributions.items():
        if group not in seen:
            schema.append((group, leaves))

    return schema


# ── Generic fallback leaves for the open file ──────────────────────────────
# (group, leaf label) → generated plugin class.  Rebuilt on every file open
# by register_generic_leaves(); hand-written REGISTRY entries always win.
GENERIC: dict[tuple[str, str], type[BasePlugin]] = {}


def register_generic_leaves(datasets: dict) -> dict[str, list[str]]:
    """
    datasets: {group: {dataset path: (shape, dtype, attrs)}} from chunk 0.
    Creates generic plugin classes for every numeric dataset without a
    registered plugin.  Returns {group: [leaf labels]} for the tree.
    """
    GENERIC.clear()
    leaves: dict[str, list[str]] = {}
    for group, members in datasets.items():
        if group in SKIP_GROUPS:
            continue
        for ds, (shape, dtype, attrs) in members.items():
            if (group, ds) in REGISTRY:
                continue
            for label, cls in make_generic_leaves(group, ds, shape, dtype).items():
                configure_units(cls, attrs)
                GENERIC[(group, label)] = cls
                leaves.setdefault(group, []).append(label)
    return leaves


def resolve_plugin_class(group: str, leaf: str) -> type[BasePlugin] | None:
    """Hand-written plugin if registered, else the generic class, else None."""
    return REGISTRY.get((group, leaf)) or GENERIC.get((group, leaf))


def get_plugin(group: str, dataset: str) -> BasePlugin | None:
    """
    Return a NEW plugin instance for (group, dataset), or None if unknown.

    Callers are responsible for caching if state persistence is required.
    CosmoExplorer._plugin_cache handles this for the main HDF5 pipeline;
    tests and diagnostic overrides call this directly to obtain intentionally
    stateless instances.

    Do NOT call this in hot paths (tab/scale/unit re-renders) — use the
    cached instance to preserve _last_bin_count hysteresis across renders.
    """
    cls = resolve_plugin_class(group, dataset)
    return cls() if cls else None


def all_required_paths() -> set[str]:
    """Return the union of all REQUIRES paths across all plugins."""
    paths: set[str] = set()
    for cls in REGISTRY.values():
        paths.update(cls.REQUIRES)
    return paths