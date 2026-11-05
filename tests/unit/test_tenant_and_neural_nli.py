"""
Unit and integration tests for Multi-Tenant Isolation, User Authentication,
Process-Isolated Tool Execution, and Neural NLI Entailment.
"""

import pytest
from api.auth import UserAuthManager
from api.tenant_manager import TenantKnowledgeBaseManager
from src.tools.implementations import SparseRetrievalTool, DenseRetrievalTool
from src.verification.neural_nli import NeuralNLIClassifier


def test_user_registration_and_authentication(tmp_path):
    auth_file = tmp_path / "test_users.json"
    mgr = UserAuthManager(storage_path=str(auth_file))

    # Register user
    user = mgr.register_user("researcher_alice", "secure_pass_123", tenant_id="tenant_alice")
    assert user.username == "researcher_alice"
    assert user.tenant_id == "tenant_alice"
    assert user.password_hash != "secure_pass_123"  # Salted hash

    # Duplicate username should raise ValueError
    with pytest.raises(ValueError):
        mgr.register_user("researcher_alice", "another_pass")

    # Correct authentication
    auth_user = mgr.authenticate_user("researcher_alice", "secure_pass_123")
    assert auth_user is not None
    assert auth_user.user_id == user.user_id

    # Wrong password should fail
    assert mgr.authenticate_user("researcher_alice", "wrong_password") is None

    # Issue session
    sess = mgr.create_session(auth_user)
    assert "session_token" in sess
    assert sess["tenant_id"] == "tenant_alice"

    # Validate session
    val = mgr.validate_session(sess["session_token"])
    assert val is not None
    assert val["username"] == "researcher_alice"


def test_guest_sandbox_session(tmp_path):
    mgr = UserAuthManager(storage_path=str(tmp_path / "guest_users.json"))
    g1 = mgr.create_guest_sandbox_session()
    g2 = mgr.create_guest_sandbox_session()

    # Sandboxes must have unique tokens and unique tenant IDs
    assert g1["session_token"] != g2["session_token"]
    assert g1["tenant_id"] != g2["tenant_id"]
    assert g1["tenant_id"].startswith("guest_")


def test_multi_tenant_knowledge_base_isolation(tmp_path):
    tm = TenantKnowledgeBaseManager(base_data_dir=str(tmp_path))

    kb_alpha = tm.get_kb("tenant_alpha")
    kb_beta = tm.get_kb("tenant_beta")

    # Ingest document into Tenant Alpha
    kb_alpha.index_document("doc_alpha_exclusive", "Alpha confidential system architecture.")

    # Ingest document into Tenant Beta
    kb_beta.index_document("doc_beta_exclusive", "Beta proprietary database design.")

    # Verify complete isolation: Alpha must not see Beta docs, Beta must not see Alpha docs
    alpha_docs = [d["doc_id"] for d in kb_alpha.get_document_summary()]
    beta_docs = [d["doc_id"] for d in kb_beta.get_document_summary()]

    assert "doc_alpha_exclusive" in alpha_docs
    assert "doc_beta_exclusive" not in alpha_docs

    assert "doc_beta_exclusive" in beta_docs
    assert "doc_alpha_exclusive" not in beta_docs

    # Search in Alpha should not retrieve Beta content
    alpha_hits = kb_alpha.sparse.search("proprietary")
    assert not any(h.doc_id == "doc_beta_exclusive" for h in alpha_hits)


def test_neural_nli_entailment_and_contradiction():
    clf = NeuralNLIClassifier.get_instance()
    premise = "Raft uses randomized election timeouts between 150ms and 300ms to elect a cluster leader."

    # Entailment test
    entail_claim = "Raft utilizes election timeouts to choose a cluster leader."
    res_entail = clf.classify(premise, entail_claim)
    assert res_entail["is_entailed"] is True
    assert res_entail["contradiction"] is False

    # Contradiction test
    contra_claim = "Raft does not use election timeouts."
    res_contra = clf.classify(premise, contra_claim)
    assert res_contra["contradiction"] is True
    assert res_contra["is_entailed"] is False


def test_process_isolated_tool_execution(tmp_path):
    tm = TenantKnowledgeBaseManager(base_data_dir=str(tmp_path))
    kb = tm.get_kb("tenant_test")
    kb.index_document("doc_proc", "Process isolation guarantees clean termination of worker tasks.")

    tool = SparseRetrievalTool(kb)

    # Normal execution
    res_std = tool.run(query="guarantees")
    assert res_std["success"] is True

    # Process-isolated execution
    res_iso = tool.run_isolated(query="guarantees")
    assert res_iso["success"] is True
    assert len(res_iso["data"]) > 0
    assert res_iso["data"][0].doc_id == "doc_proc"
