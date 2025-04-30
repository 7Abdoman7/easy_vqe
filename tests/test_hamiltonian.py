import pytest
import numpy as np
from easy_vqe.hamiltonian import parse_hamiltonian_expression, get_theoretical_ground_state_energy

# === Tests for parse_hamiltonian_expression ===

def test_parse_valid_simple():
    """Test basic valid Hamiltonian parsing."""
    h_str = "1.0 * XX - 0.5 * ZI + YZ"
    expected = [(1.0, "XX"), (-0.5, "ZI"), (1.0, "YZ")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == len(expected)
    for p, e in zip(parsed, expected):
        assert np.isclose(p[0], e[0])
        assert p[1] == e[1]

def test_parse_valid_implicit_coeffs():
    """Test parsing with implicit +/- 1 coefficients."""
    h_str = "XX + YY - ZZ + I" # Single qubit implicit
    expected = [(1.0, "XX"), (1.0, "YY"), (-1.0, "ZZ"), (1.0, "I")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == len(expected)
    assert parsed[0] == (1.0, "XX")
    assert parsed[1] == (1.0, "YY")
    assert parsed[2] == (-1.0, "ZZ")
    assert parsed[3] == (1.0, "I") # Check single qubit case

def test_parse_valid_mixed_coeffs_spacing():
    """Test parsing with mixed coefficients and varied spacing."""
    h_str = "- ZI   +  3.14 * XY -1.0*ZZ   +II"
    expected = [(-1.0, "ZI"), (3.14, "XY"), (-1.0, "ZZ"), (1.0, "II")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == len(expected)
    assert np.isclose(parsed[0][0], expected[0][0]) and parsed[0][1] == expected[0][1]
    assert np.isclose(parsed[1][0], expected[1][0]) and parsed[1][1] == expected[1][1]
    assert np.isclose(parsed[2][0], expected[2][0]) and parsed[2][1] == expected[2][1]
    assert np.isclose(parsed[3][0], expected[3][0]) and parsed[3][1] == expected[3][1]

def test_parse_valid_leading_term_no_sign():
    """Test parsing when the first term has no leading sign."""
    h_str = "0.5 * ZZ - IX"
    expected = [(0.5, "ZZ"), (-1.0, "IX")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == len(expected)
    assert np.isclose(parsed[0][0], expected[0][0]) and parsed[0][1] == expected[0][1]
    assert np.isclose(parsed[1][0], expected[1][0]) and parsed[1][1] == expected[1][1]

    h_str_2 = "XYZ + 0.1*III"
    expected_2 = [(1.0, "XYZ"), (0.1, "III")]
    parsed_2 = parse_hamiltonian_expression(h_str_2)
    assert len(parsed_2) == len(expected_2)
    assert np.isclose(parsed_2[0][0], expected_2[0][0]) and parsed_2[0][1] == expected_2[0][1]
    assert np.isclose(parsed_2[1][0], expected_2[1][0]) and parsed_2[1][1] == expected_2[1][1]


def test_parse_valid_scientific_notation():
    """Test parsing coefficients in scientific notation."""
    h_str = "1e-3 * X - 2.5E+2 * Y + 1.23e1 * Z"
    expected = [(0.001, "X"), (-250.0, "Y"), (12.3, "Z")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == len(expected)
    assert np.isclose(parsed[0][0], expected[0][0])
    assert parsed[0][1] == expected[0][1]
    assert np.isclose(parsed[1][0], expected[1][0])
    assert parsed[1][1] == expected[1][1]
    assert np.isclose(parsed[2][0], expected[2][0])
    assert parsed[2][1] == expected[2][1]

def test_parse_valid_combined_signs():
    """Test parsing coefficients like '+ -1.5 * X'."""
    h_str = "+ -1.5 * X - +0.5 * Y" # This syntax is slightly odd but should parse
    expected = [(-1.5, "X"), (-0.5, "Y")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == len(expected)
    assert np.isclose(parsed[0][0], expected[0][0])
    assert parsed[0][1] == expected[0][1]
    assert np.isclose(parsed[1][0], expected[1][0])
    assert parsed[1][1] == expected[1][1]

def test_parse_error_invalid_pauli_char():
    """Test ValueError for invalid characters in Pauli string."""
    with pytest.raises(ValueError, match=r"Invalid character.*'A'.*'XA'"):
        parse_hamiltonian_expression("1.0 * XA")
    with pytest.raises(ValueError, match=r"Invalid character.*'B'.*'YYB'"):
        parse_hamiltonian_expression("YYB")

def test_parse_error_inconsistent_length():
    """Test ValueError for inconsistent Pauli string lengths."""
    with pytest.raises(ValueError, match=r"Inconsistent Pauli string lengths.*Term 0 'XX'.*Term 1 'YYY'"):
        parse_hamiltonian_expression("1.0 * XX + 0.5 * YYY")
    with pytest.raises(ValueError, match=r"Inconsistent Pauli string lengths.*Term 0 'Z'.*Term 1 'II'"):
        parse_hamiltonian_expression("Z + II")

def test_parse_error_invalid_syntax():
    """Test ValueError for various syntax errors."""
    with pytest.raises(ValueError, match=r"Could not parse term.*'\+\+ 0.5 \* YY'"):
        parse_hamiltonian_expression("1.0 * XX ++ 0.5 * YY")
    with pytest.raises(ValueError, match=r"Could not parse term.*'\*$'"): # Matches '*' at the end
        parse_hamiltonian_expression("1.0 *")
    with pytest.raises(ValueError, match=r"Ambiguous term 'YY'"): # Term after first needs sign or coeff
        parse_hamiltonian_expression("XX YY")
    with pytest.raises(ValueError, match=r"Invalid numeric coefficient '1.0a'"):
        parse_hamiltonian_expression("1.0a * X")
    with pytest.raises(ValueError, match=r"Invalid coefficient value \(nan\)"):
         parse_hamiltonian_expression("nan * X")
    with pytest.raises(ValueError, match=r"Invalid coefficient value \(inf\)"):
         parse_hamiltonian_expression("inf * Z")
    with pytest.raises(ValueError, match=r"Unexpected '\*' without preceding coefficient"):
         parse_hamiltonian_expression("* XY")
    with pytest.raises(ValueError, match=r"Could not parse term.*'1\.0 Z'"): # Missing '*'
        parse_hamiltonian_expression("1.0 Z")
    with pytest.raises(ValueError, match=r"Could not parse any valid.*empty string"):
        parse_hamiltonian_expression(" ")
    with pytest.raises(ValueError, match=r"Empty Pauli string found"):
        parse_hamiltonian_expression("1.0 * ")

def test_parse_error_empty_string():
    """Test ValueError for empty input string."""
    with pytest.raises(ValueError, match="Hamiltonian expression cannot be empty."):
        parse_hamiltonian_expression("")

def test_parse_error_non_string_input():
    """Test TypeError for non-string input."""
    with pytest.raises(TypeError, match="Hamiltonian expression must be a string."):
        parse_hamiltonian_expression(123)
    with pytest.raises(TypeError, match="Hamiltonian expression must be a string."):
        parse_hamiltonian_expression(["1.0 * X"])

def test_parse_zero_coefficients():
    """Test that terms with zero coefficients are still parsed."""
    h_str = "0.0 * XX + 1.0 * YY"
    expected = [(0.0, "XX"), (1.0, "YY")]
    parsed = parse_hamiltonian_expression(h_str)
    assert len(parsed) == 2
    assert np.isclose(parsed[0][0], 0.0)
    assert parsed[0][1] == "XX"
    assert np.isclose(parsed[1][0], 1.0)
    assert parsed[1][1] == "YY"

# === Tests for get_theoretical_ground_state_energy ===

@pytest.mark.parametrize("h_str, expected_energy", [
    ("Z", -1.0),          # Single qubit Z
    ("X", -1.0),          # Single qubit X
    ("-1.5 * Z", -1.5),   # Scaled Z
    ("0.5 * X", -0.5),    # Scaled X
    ("Z + X", -np.sqrt(2)), # Z+X -> eigenvalues are +/- sqrt(1^2+1^2)
    ("ZZ", -1.0),         # Two qubit ZZ
    ("XX + ZZ", -2.0),    # Bell state ground energy for XX+ZZ
    ("1.0 * II + 0 * ZI", 1.0), # Identity plus zero term
    ("- Z", 1.0),         # Minus Z, ground state is |1> with energy -(-1)=1
    ("0.5*XI - 0.5*IX + 0.2*ZZ", -0.73851648), # More complex 2 qubit
])
def test_ground_state_energy_correct(h_str, expected_energy):
    """Test theoretical ground state energy calculation for known cases."""
    calculated_energy = get_theoretical_ground_state_energy(h_str)
    assert np.isclose(calculated_energy, expected_energy, atol=1e-7)

def test_ground_state_energy_invalid_hamiltonian():
    """Test that ground state energy calculation raises error on invalid input."""
    with pytest.raises(ValueError):
        get_theoretical_ground_state_energy("1.0 * XX + YYY") # Inconsistent length
    with pytest.raises(ValueError):
        get_theoretical_ground_state_energy("1.0 * ZA") # Invalid char