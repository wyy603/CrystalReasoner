"""Round-trip tests for crysreas.mlip.cte _serialize / _deserialize (Ray-safe payloads)."""

from __future__ import annotations

import numpy as np
import pytest
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Lattice, Structure

from crysreas.mlip.cte import (
    ConfigQHA,
    _collect_supercell,
    _deserialize,
    _qha_task_from_wire,
    _serialize,
)


def hello_world():
    print("hello_world")

def test_structure_round_trip():
    lattice = Lattice.cubic(5.43)
    struct = Structure(lattice=lattice, species=["Si", "Si"], coords=[[0, 0, 0], [0.25, 0.25, 0.25]])
    payload = _serialize(struct)
    assert payload["_dtype"] == "pymatgen.core.structure.Structure"
    back = _deserialize(payload, Structure)
    assert np.allclose(struct.lattice.matrix, back.lattice.matrix)
    assert len(struct) == len(back)
    for a, b in zip(struct, back):
        assert str(a.specie) == str(b.specie)
        assert np.allclose(a.frac_coords, b.frac_coords)


def test_phonopy_atoms_round_trip():
    pa = PhonopyAtoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]],
        cell=np.eye(3, dtype=np.float64) * 5.43,
    )
    payload = _serialize(pa)
    assert payload["_dtype"] == "phonopy.structure.atoms.PhonopyAtoms"
    back = _deserialize(payload, PhonopyAtoms)
    assert pa.symbols == back.symbols
    assert np.allclose(pa.cell, back.cell)
    assert np.allclose(pa.positions, back.positions)

def test_phonopy_round_trip_after_collect_supercell():
    lattice = Lattice.cubic(5.43)
    struct = Structure(lattice=lattice, species=["Si"], coords=[[0, 0, 0]])
    cfg = ConfigQHA()
    ph_orig, scs_orig = _collect_supercell(struct, cfg.model_dump())
    payload = _serialize(ph_orig)
    assert payload["_dtype"] == "phonopy.api_phonopy.Phonopy"
    assert payload["displacement_distance"] is not None

    ph_back = _deserialize(payload, Phonopy)
    scs_back = ph_back.supercells_with_displacements
    assert scs_back is not None
    assert len(scs_orig) == len(scs_back)

    n_atoms = len(ph_orig.supercell)
    for a, b in zip(scs_orig, scs_back):
        assert np.allclose(a.positions, b.positions, atol=1e-9)
        assert np.allclose(a.cell, b.cell, atol=1e-9)
        assert list(a.symbols) == list(b.symbols)
        assert len(a.positions) == n_atoms


def test_deserialize_type_mismatch_raises():
    lattice = Lattice.cubic(3.0)
    struct = Structure(lattice=lattice, species=["Si"], coords=[[0, 0, 0]])
    payload = _serialize(struct)
    with pytest.raises(TypeError, match="expected.*PhonopyAtoms"):
        _deserialize(payload, PhonopyAtoms)


def test_serialize_rejects_unsupported_type():
    with pytest.raises(TypeError, match="unsupported"):
        _serialize({1, 2})  # type: ignore[arg-type]


def test_primitive_and_container_round_trip():
    assert _deserialize(_serialize(None)) is None
    assert _deserialize(_serialize(True)) is True
    assert _deserialize(_serialize(False)) is False
    assert _deserialize(_serialize(42)) == 42
    assert _deserialize(_serialize(3.25)) == 3.25
    assert _deserialize(_serialize("abc")) == "abc"
    nested = {"a": [1, 2, (3, 4)], "b": np.arange(3)}
    back = _deserialize(_serialize(nested))
    assert back["a"][2] == (3, 4)
    assert np.allclose(back["b"], [0, 1, 2])

def test_numpy_scalar_round_trip():
    assert _deserialize(_serialize(np.int64(7))) == 7
    assert _deserialize(_serialize(np.float64(1.5))) == 1.5


def test_qha_task_tuple_from_wire():
    lattice = Lattice.cubic(5.43)
    struct = Structure(lattice=lattice, species=["Si"], coords=[[0, 0, 0]])
    cfg = ConfigQHA()
    ph, _scs = _collect_supercell(struct, cfg.model_dump())
    task = (
        [ph],
        [struct],
        [0.1],
        [[]],
        np.zeros((2, 3)),
        np.array([0, 100, 200]),
        [20, 20, 20],
        10,
        0,
    )
    wire = _serialize(task)
    restored = _qha_task_from_wire(wire)
    assert len(restored) == 9
    assert isinstance(restored[4], np.ndarray)
    assert isinstance(restored[5], np.ndarray)
    assert restored[7] == 10 and restored[8] == 0
