"""Text snippets for electrical, stability, and mechanical properties."""


def electrical_info_prompt(elem):
    is_metal = elem.get("is_metal", False)
    band_gap = elem.get("band_gap")
    efermi = elem.get("efermi")

    if is_metal and efermi is not None:
        electrical_info = (
            f"This material is classified as a metal, confirmed by its zero band gap "
            f"and a Fermi energy ($E_F$) of {efermi:.2f} eV."
        )
    elif band_gap is not None and band_gap > 0.0:
        # Define insulator/semiconductor based on the 3.0 eV threshold
        band_gap_type = (
            "insulator (wide band gap)" if band_gap > 3.0 else "semiconductor (narrow/moderate band gap)"
        )
        electrical_info = f"It is an {band_gap_type} with a calculated band gap ($E_g$) of {band_gap:.3f} eV."
    else:
        electrical_info = ""
    return electrical_info


def stability_status_prompt(elem):
    is_stable = elem.get("is_stable", False)
    energy_above_hull = elem.get("energy_above_hull")
    decomposition_list = elem.get("decomposes_to")
    formation_energy = elem.get("formation_energy_per_atom")

    stability_status = ""
    if is_stable:
        stability_status = "It is predicted to be thermodynamically stable (on the hull)."
    else:
        stability_status = (
            f"It is metastable, lying {energy_above_hull:.3f} eV/atom above the convex hull. "
        )
    if formation_energy is not None:
        stability_status += f" The formation energy per atom is {formation_energy:.3f} eV/atom."
    return stability_status


def mechanical_summary_prompt(elem):
    """Generate a textual summary of the material's mechanical properties."""
    bulk_modulus_data = elem.get("bulk_modulus")
    shear_modulus_data = elem.get("shear_modulus")
    poisson = elem.get("homogeneous_poisson")

    mechanical_info_list = []
    anisotropy_comment = ""

    # 1. Process Bulk Modulus
    if bulk_modulus_data is not None and isinstance(bulk_modulus_data, dict):
        K_vrh = bulk_modulus_data.get("vrh")
        if K_vrh is not None:
            mechanical_info_list.append(f"a Bulk Modulus ($K_{{VRH}}$) of {K_vrh:.3f} GPa")
    elif isinstance(bulk_modulus_data, (int, float)):
        mechanical_info_list.append(f"a Bulk Modulus ($K$) of {bulk_modulus_data:.3f} GPa")

    # 2. Process Shear Modulus and calculate anisotropy
    if shear_modulus_data is not None and isinstance(shear_modulus_data, dict):
        G_vrh = shear_modulus_data.get("vrh")
        G_voigt = shear_modulus_data.get("voigt")
        G_reuss = shear_modulus_data.get("reuss")

        if G_vrh is not None:
            mechanical_info_list.append(f"a Shear Modulus ($G_{{VRH}}$) of {G_vrh:.3f} GPa")

            if G_voigt is not None and G_reuss is not None:
                # Calculate percentage anisotropy index
                percent_difference = abs(G_voigt - G_reuss) / G_vrh * 100

                if percent_difference < 1.0:  # Isotropic if difference < 1%
                    anisotropy_comment = "The material exhibits near-isotropic shear behavior."
                elif percent_difference >= 1.0:
                    anisotropy_comment = (
                        f"Due to anisotropy, the shear response ranges from {G_reuss:.3f} GPa "
                        f"(Reuss, lower bound) to {G_voigt:.3f} GPa (Voigt, upper bound)."
                    )

    elif isinstance(shear_modulus_data, (int, float)):
        mechanical_info_list.append(f"a Shear Modulus ($G$) of {shear_modulus_data:.3f} GPa")

    # 3. Process Poisson Ratio
    if poisson is not None:
        mechanical_info_list.append(f"a homogeneous Poisson ratio ($\\nu$) of {poisson:.3f}")

    # 4. Final summary
    if not mechanical_info_list:
        return ""
    mechanical_summary = "It exhibits " + ", and ".join(mechanical_info_list) + "."

    if anisotropy_comment:
        mechanical_summary += f" {anisotropy_comment}"

    return mechanical_summary
