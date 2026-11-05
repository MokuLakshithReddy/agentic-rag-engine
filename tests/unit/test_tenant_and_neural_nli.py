"""
Unit and integration tests for Multi-Tenant Isolation, User Authentication,
Bootstrap Admin Security, Path Traversal Guards, Process-Isolated Execution,
and Neural NLI Explicit Logging and Verification.
"""

import os
import time
import pytest
from pathlib import Path
from api.auth import UserAuthManager
from api.tenant_manager import TenantKnowledgeBaseManager
from src.tools.base import BaseTool
from src.tools.implementations import SparseRetrievalTool, DenseRetrievalTool
from src.tools.registry import ToolRegistry
from src.agents.workflow import AgenticRAGWorkflow
from src.verification.neural_nli import NeuralNLIClassifier
from src.verification.verifier import AnswerVerifier
from src.state.state import AgentState, EvidenceChunk, ResponseStatus


def test_user_registration_server_generated_tenants_and_auth(tmp_path):
    auth_file = tmp_path / "test_users.json"
    tenants_file = tmp_path / "test_tenants.json"
    mgr = UserAuthManager(storage_path=str(auth_file), tenants_path=str(tenants_file))

    # Register user without arbitrary tenant_id -> server creates unguessable tenant_id
    user = mgr.register_user("researcher_alice", "secure_pass_123")
    assert user.username == "researcher_alice"
    assert user.tenant_id.startswith("tenant_")
    assert user.password_hash != "secure_pass_123"  # Salted PBKDF2 hash

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
    assert sess["tenant_id"] == user.tenant_id

    # Validate session
    val = mgr.validate_session(sess["session_token"])
    assert val is not None
    assert val["username"] == "researcher_alice"


def test_tenant_invitation_code_membership(tmp_path):
    auth_file = tmp_path / "test_users.json"
    tenants_file = tmp_path / "test_tenants.json"
    mgr = UserAuthManager(storage_path=str(auth_file), tenants_path=str(tenants_file))

    # Alice creates an organization
    alice = mgr.register_user("alice", "alice_pass_123")
    alice_tenant = alice.tenant_id

    # Alice creates an invitation code
    code = mgr.create_invitation(tenant_id=alice_tenant, creator_user_id=alice.user_id, role="user")
    assert code.startswith("inv_")

    # Bob registers using the invitation code
    bob = mgr.register_user("bob", "bob_pass_123", invitation_code=code)
    assert bob.tenant_id == alice_tenant  # Joined Alice's organization

    # Using the same invitation code again fails
    with pytest.raises(ValueError, match="Invalid or expired"):
        mgr.register_user("charlie", "charlie_pass_123", invitation_code=code)


def test_admin_bootstrap_secret_and_password_replacement(tmp_path, monkeypatch):
    # Ensure ADMIN_PASSWORD is unset so bootstrap flow triggers
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    auth_file = tmp_path / "data" / "users.json"
    secret_file = tmp_path / "data" / ".bootstrap_admin_secret"

    mgr = UserAuthManager(storage_path=str(auth_file))

    # Secret file should have been generated
    assert secret_file.exists()
    bootstrap_secret = secret_file.read_text(encoding="utf-8").strip()
    assert len(bootstrap_secret) >= 32

    # Admin user should have must_change_password=True
    admin_auth = mgr.authenticate_user("admin", bootstrap_secret)
    assert admin_auth is not None
    assert admin_auth.must_change_password is True

    # Password change required
    mgr.change_password(
        user_id=admin_auth.user_id,
        current_password=bootstrap_secret,
        new_password="ReplacedAdminPassword2026!"
    )

    # Subsequent authentication with new password has must_change_password=False
    updated_admin = mgr.authenticate_user("admin", "ReplacedAdminPassword2026!")
    assert updated_admin is not None
    assert updated_admin.must_change_password is False


def test_tenant_path_traversal_guards(tmp_path):
    tm = TenantKnowledgeBaseManager(base_data_dir=str(tmp_path))

    # Valid tenant IDs pass
    kb = tm.get_kb("tenant_valid_123")
    assert kb is not None

    # Path traversal attempts must raise ValueError
    with pytest.raises(ValueError, match="Invalid tenant"):
        tm.get_kb("../evil")

    with pytest.raises(ValueError, match="Invalid tenant"):
        tm.get_kb("tenant/../../etc")

    with pytest.raises(ValueError, match="Invalid tenant"):
        tm.get_kb("tenant..breakout")

    with pytest.raises(ValueError, match="Invalid tenant"):
        tm.get_kb("..\\windows\\breakout")


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


class HangingRetrievalTool(BaseTool):
    """Mock retrieval tool simulating an uncooperative hanging worker."""
    def __init__(self, timeout_sec: float = 0.5, process_isolated: bool = True):
        super().__init__(
            name="hanging_retrieval",
            description="Simulates a hanging retrieval operation.",
            timeout_sec=timeout_sec,
            process_isolated=process_isolated
        )

    def _execute(self, **kwargs):
        # Sleep far longer than timeout_sec
        time.sleep(3.0)
        return []


def test_process_isolated_tool_hard_timeout_termination():
    tool = HangingRetrievalTool(timeout_sec=0.5, process_isolated=True)
    start_t = time.time()
    res = tool.run(query="test_hang")
    elapsed = time.time() - start_t

    # Must complete within bounded time (well under 3.0s) and report timeout failure
    assert elapsed < 2.0
    assert res["success"] is False
    assert "timeout" in res["error"].lower() or "hard-killed" in res["error"].lower()


def test_verifier_deterministic_fallback_and_reporting(monkeypatch):
    """
    Base installation test:
    Verifies that when optional neural ML dependencies or model weights are unavailable,
    the deterministic fallback executes reliably and is explicitly reported in metadata.
    Does not depend on undeclared ML packages or external network downloads.
    """
    clf = NeuralNLIClassifier.get_instance()
    # Explicitly simulate base/offline environment without neural weights
    monkeypatch.setattr(clf, "_is_ready", False)
    monkeypatch.setattr(clf, "model", None)

    verifier = AnswerVerifier(verification_strategy="auto")
    evidence = EvidenceChunk(
        chunk_id="raft_paper#chunk_raft_01",
        doc_id="raft_paper",
        content="Raft accomplishes consensus via an elected leader who manages log replication."
    )
    state = AgentState(
        original_query="How does Raft manage log replication?",
        retrieved_evidence=[evidence],
        evidence_ids=["raft_paper#chunk_raft_01"],
        evidence_relevance_score=0.95
    )
    state.synthesis_draft = "Raft accomplishes consensus through an elected leader managing log replication [raft_paper#chunk_raft_01]."

    _, confidence, status = verifier.verify(state)
    assert status == ResponseStatus.SUCCESS
    assert state.verification_metadata.get("strategy") == "auto"
    assert state.verification_metadata.get("model_used") == "deterministic_lexical_fallback"
    assert state.verification_metadata.get("claims_evaluated") >= 1


def test_heuristic_fallback_entailment_and_contradiction():
    """
    Base installation test:
    Validates calibrated deterministic NLI fallback rules when neural weights are absent.
    """
    clf = NeuralNLIClassifier.get_instance()
    premise = "Raft uses randomized election timeouts between 150ms and 300ms to elect a cluster leader."

    # Entailment fallback test
    entail_claim = "Raft utilizes election timeouts to choose a cluster leader."
    res_entail = clf._heuristic_fallback(premise, entail_claim)
    assert res_entail["is_entailed"] is True
    assert res_entail["contradiction"] is False

    # Contradiction fallback test
    contra_claim = "Raft does not use election timeouts."
    res_contra = clf._heuristic_fallback(premise, contra_claim)
    assert res_contra["contradiction"] is True
    assert res_contra["is_entailed"] is False


@pytest.mark.skipif(
    not getattr(NeuralNLIClassifier.get_instance(), "_is_ready", False),
    reason="Requires optional 'ml' dependency group (sentence-transformers) and downloaded model weights"
)
def test_neural_nli_model_inference_and_metadata_logging():
    """
    Neural installation test:
    Verifies that when the optional 'ml' dependency group is installed, the DeBERTa
    cross-encoder model loads, performs semantic inference, and reports neural metadata.
    """
    clf = NeuralNLIClassifier.get_instance()
    assert clf._is_ready is True
    assert clf.model is not None

    premise = "Raft uses randomized election timeouts between 150ms and 300ms to elect a cluster leader."
    entail_claim = "Raft utilizes election timeouts to choose a cluster leader."
    res_entail = clf.classify(premise, entail_claim)
    assert res_entail["is_entailed"] is True
    assert res_entail["contradiction"] is False

    contra_claim = "Raft does not use election timeouts."
    res_contra = clf.classify(premise, contra_claim)
    assert res_contra["contradiction"] is True
    assert res_contra["is_entailed"] is False

    verifier = AnswerVerifier(verification_strategy="neural_nli")
    evidence = EvidenceChunk(
        chunk_id="raft_paper#chunk_raft_01",
        doc_id="raft_paper",
        content="Raft accomplishes consensus via an elected leader who manages log replication."
    )
    state = AgentState(
        original_query="How does Raft manage log replication?",
        retrieved_evidence=[evidence],
        evidence_ids=["raft_paper#chunk_raft_01"],
        evidence_relevance_score=0.95
    )
    state.synthesis_draft = "Raft accomplishes consensus through an elected leader managing log replication [raft_paper#chunk_raft_01]."

    _, confidence, status = verifier.verify(state)
    assert status == ResponseStatus.SUCCESS
    assert confidence > 0.60
    assert state.verification_metadata.get("strategy") == "neural_nli"
    assert state.verification_metadata.get("model_used") == "cross-encoder/nli-deberta-v3-small"
