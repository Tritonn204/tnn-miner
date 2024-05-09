package main

import (
    "fmt"
    "math/big"
)

func main() {
    difficulty := big.NewInt(12)

    // Create a big.Int for (1<<256)-1
    maxTarget := new(big.Int)
    maxTarget.Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))

    // Calculate the target value using the difficulty
    target := new(big.Int)
    target.Div(maxTarget, difficulty)

    // Calculate the mantissa and exponent
    mantissa := new(big.Int).Set(target)
    exponent := uint(len(target.Bytes()))

    // Adjust the mantissa and exponent
    if exponent <= 3 {
        mantissa.Rsh(mantissa, 8*(3-exponent))
    } else {
        mantissa.Lsh(mantissa, 8*(exponent-3))
    }

    fmt.Printf("Difficulty: %v\n", difficulty)
    fmt.Printf("Target: %v\n", target)
    fmt.Printf("Mantissa: %v\n", mantissa)
    fmt.Printf("Exponent: %v\n", exponent)
}