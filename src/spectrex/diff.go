package main

import (
    "fmt"
    "math/big"
)

var (
	maxTarget = big.NewFloat(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
	minHash   = new(big.Float).Quo(new(big.Float).SetMantExp(big.NewFloat(1), 256), maxTarget)
	bigGig    = big.NewFloat(1e3)
)

func DiffToTarget(diff float64) *big.Int {
    target := new(big.Float).Quo(maxTarget, big.NewFloat(diff))

    t, _ := target.Int(nil)
    return t
}

func DiffToHash(diff float64) float64 {
	hashVal := new(big.Float).Mul(minHash, big.NewFloat(diff))
	hashVal.Quo(hashVal, bigGig)

	h, _ := hashVal.Float64()
	return h
}

func main() {
    difficulty := float64(12)

    target := DiffToTarget(difficulty)
		hash := DiffToHash(difficulty)

    fmt.Printf("Difficulty: %f\n", difficulty)
    fmt.Printf("Target: %v\n", target)
		fmt.Printf("Target (HEX): %x\n", target)
		fmt.Printf("Hash: %v\n", hash)
}