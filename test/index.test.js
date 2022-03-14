const assert = require('assert')

const arbp = require('..')


describe('Testing', () => {
    it('Test pseudo normal random value', () => {
        const e = arbp.pseudoNormal()
    })
    it('Test cumMin', () => {
        const z = [2, 1, 2, 1.1, 2, 1]
        const m = arbp.cumMin(z)
        assert.deepEqual(m, [2, 1, 1, 1, 1, 1])
    })
    it('Test ARModel', () => {
        // Time series
        const z = [1, 2, 1.1, 2, 1, 2]

        // Create a model object
        const model = new arbp.ARModel()

        // Train the model
        model.fit(z)

        // Predict next 10 values randomly via the model
        const h = model.simulate(z, 10)
        console.log(h)

        // Create border predictor instance with 100 simulations
        const bp = new arbp.BorderPredictor(100)

        // Simulate next 100 values for each trajectory with the trained model
        bp.simulate(z, 100, model)

        // Predict the border with the probability 0.9 on the step 50
        const border = bp.predict(0.9, 50)
        console.log(border)
    })
})
