// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {PlonkVerifier} from "./PlonkVerifier.sol";

/// @title ZKLRBatchVerifier
/// @notice Verifies ZKLR batch prediction proofs against a pinned model.
///
/// A ZKLR proof states: "the model committed to in `modelCommitment` predicts
/// label L_i for features X_i, for every sample i in the batch". The PLONK
/// verifier checks the proof against the public inputs; this wrapper
/// additionally pins the model commitment so provers cannot substitute a
/// different model after deployment.
///
/// Public input layout (BatchLabelCircuit):
///   [ X[0][0..F-1], ..., X[B-1][0..F-1], Label[0..B-1], Commitment ]
contract ZKLRBatchVerifier is PlonkVerifier {
    /// @notice MiMC(W..., B) of the model this contract accepts proofs for.
    uint256 public immutable modelCommitment;

    error WrongModelCommitment(uint256 got, uint256 want);
    error EmptyPublicInputs();

    constructor(uint256 _modelCommitment) {
        modelCommitment = _modelCommitment;
    }

    /// @notice Verify a batch prediction proof for the pinned model.
    /// @param proof PLONK proof, serialized with gnark's MarshalSolidity
    /// @param publicInputs features, labels and commitment (see layout above)
    /// @return success true iff the proof is valid for the pinned model
    function verifyBatch(bytes calldata proof, uint256[] calldata publicInputs)
        external
        view
        returns (bool success)
    {
        if (publicInputs.length == 0) revert EmptyPublicInputs();
        uint256 got = publicInputs[publicInputs.length - 1];
        if (got != modelCommitment) {
            revert WrongModelCommitment(got, modelCommitment);
        }
        return this.Verify(proof, publicInputs);
    }
}
