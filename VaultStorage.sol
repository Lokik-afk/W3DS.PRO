// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VaultStorage {
    address public owner;
    string public username;
    uint256 public createdAt;

    constructor(string memory _username) {
        owner = msg.sender;
        username = _username;
        createdAt = block.timestamp;
    }

    function getVaultInfo() public view returns (address, string memory, uint256) {
        return (owner, username, createdAt);
    }
}
