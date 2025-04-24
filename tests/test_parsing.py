import pytest
import numpy as np

from easy_vqe import parse_hamiltonian_expression

def test_simple_parsing():
    """Test basic Hamiltonian parsing."""
    h_str = "1.0 * XX - 0.5 * ZI + YZ"
    expected = [(1.0, "XX"), (-0.5, "ZI"), (1.0, "YZ")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == len(expected)
    for p, e in zip(parsed, expected):
        assert np.isclose(p[0], e[0])
        assert p[1] == e[1]

def test_implicit_coeffs():
    """Test parsing with implicit +/- 1 coefficients."""
    h_str = "XX + YY - ZZ"
    expected = [(1.0, "XX"), (1.0, "YY"), (-1.0, "ZZ")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == len(expected)
 
def test_mixed_coeffs():
    """Test parsing with mixed explicit/implicit coefficients and spacing."""
    h_str = "- ZI + 3.14* XY - 1.0*ZZ + II"
    expected = [(-1.0, "ZI"), (3.14, "XY"), (-1.0, "ZZ"), (1.0, "II")]
    parsed = parse_hamiltonian_expression(h_str)

def test_invalid_pauli():
    """Test that invalid characters raise ValueError."""
    with pytest.raises(ValueError, match=r"Invalid char.*'A'.*"):
        parse_hamiltonian_expression("1.0 * XA")

def test_inconsistent_length():
    """Test that inconsistent Pauli string lengths raise ValueError."""
    with pytest.raises(ValueError, match=r"Inconsistent Pauli string lengths.*"):
        parse_hamiltonian_expression("1.0 * XX + 0.5 * YYY")

def test_invalid_syntax():
    """Test some syntax errors."""
    with pytest.raises(ValueError): 
        parse_hamiltonian_expression("1.0 * XX ++ 0.5 * YY")
    with pytest.raises(ValueError):
        parse_hamiltonian_expression("1.0 *")
    with pytest.raises(ValueError):
        parse_hamiltonian_expression("XX YY") 

