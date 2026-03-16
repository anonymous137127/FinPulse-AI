from fastapi import APIRouter, Depends
from app.auth import require_role
from app.blockchain import blockchain

router = APIRouter()

@router.get("/verify-integrity")
def verify_integrity(
    user: dict = Depends(require_role(["admin","analyst","auditor"]))
):

    valid = blockchain.is_chain_valid()

    return {
        "status": "Valid" if valid else "Tampered"
    }