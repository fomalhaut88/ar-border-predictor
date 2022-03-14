const linear = require('linear-solve')


/**
 * Generates a random value with the normal distribution with mean = 0 
 * and variance = 1. The algorithm is approximate.
 */
function pseudoNormal() {
    let res = -6.0
    for (let i = 0; i < 12; i++) {
        res += Math.random()
    }
    return res
}


/**
 * Generated cumulative minimum for given array.
 */
function cumMin(arr) {
    if (arr.length == 0) {
        return []
    }
    let last = arr[0]
    const res = []
    for (const e of arr) {
        last = (e < last) ? e : last
        res.push(last)
    }
    return res
}


/**
 * A class that implements autoregressive (AR) model.
 */
class ARModel {
    constructor(dim) {
        if (dim === undefined) {
            dim = 1
        }
        this._dim = dim
        this._coefs = null
        this._sigma = null
    }

    fit(z) {
        // Size of the dataset to train
        const size = z.length - this._dim

        // Build zero matrix A
        const A = [...Array(this._dim + 1).keys()].map(
            i => new Array(this._dim + 1).fill(0)
        )

        // Build zero vector B
        const B = new Array(this._dim + 1).fill(0)
        
        // Accumulate A and B
        for (let i = 0; i < size; i++) {
            const yi = z[i + this._dim]
            for (let a = 0; a <= this._dim; a++) {
                const xia = (a > 0) ? z[i + this._dim - a] : 1.0 
                for (let b = 0; b <= this._dim; b++) {
                    const xib = (b > 0) ? z[i + this._dim - b] : 1.0 
                    A[a][b] += xia * xib
                }
                B[a] += xia * yi
            }
        }

        // Find coefficients as a solution of the equation: A x = B
        this._coefs = linear.solve(A, B)

        // Accumulate the error
        let E = 0.0
        for (let i = 0; i < size; i++) {
            const yi = z[i + this._dim]
            let hi = this._coefs[0]
            for (let a = 1; a <= this._dim; a++) {
                hi += this._coefs[a] * z[i + this._dim - a]
            }
            E += (yi - hi) * (yi - hi)
        }

        // Calculate standard deviation
        this._sigma = Math.sqrt(E / size)
    }

    simulate(z, size) {
        // Get last dim values
        let h = z.slice(-this._dim)

        // Generate next values randomly according to the model
        for (let i = 0; i < size; i++) {
            let h_next = this._sigma * pseudoNormal() + this._coefs[0]
            for (let a = 1; a <= this._dim; a++) {
                h_next += this._coefs[a] * h[h.length - a]
            }
            h.push(h_next)
        }

        // Return generated values only
        return h.slice(this._dim)
    }
}


/**
 * A class that implements a simulation with an AR model and 
 * prediction the minimum border.
 */
class BorderPredictor {
    constructor(simCount) {
        if (simCount === undefined) {
            simCount = 100
        }
        this._simCount = simCount
        this._size = null
        this._trajectories = null
    }

    reset() {
        this._size = null
        this._trajectories = null
    }

    simulate(z, size, model) {
        this._size = size
        this._trajectories = []

        for (let i = 0; i < this._simCount; i++) {
            const trajectory = model.simulate(z, size)
            const trajectoryCumMin = cumMin(trajectory)
            this._trajectories.push(trajectoryCumMin)
        }
    }

    predict(prob, step) {
        const arr = this._trajectories.map(t => t[step])
        arr.sort()
        const idx = Math.round(prob * this._simCount)
        return arr[idx]
    }
}


module.exports = {
    pseudoNormal, cumMin, ARModel, BorderPredictor
}
