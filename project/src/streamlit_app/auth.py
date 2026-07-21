"""Auth gate: invite register, login, ALLOW_ANON local user."""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from project.src.streamlit_app import db
from project.src.streamlit_app.session_manager import (
    SS_AUTHENTICATED,
    SS_USER_ID,
    SS_USERNAME,
)


def allow_anon() -> bool:
    load_dotenv()
    return os.getenv("ALLOW_ANON", "1").strip() in {"1", "true", "True", "yes", "YES"}


def invite_code_expected() -> str:
    load_dotenv()
    return os.getenv("CLASS_INVITE_CODE", "").strip()


def ensure_authenticated() -> bool:
    """Return True when the user may proceed into the app."""
    db.init_db()

    if st.session_state.get(SS_AUTHENTICATED) and st.session_state.get(SS_USER_ID):
        return True

    if allow_anon():
        user = db.ensure_local_user()
        st.session_state[SS_AUTHENTICATED] = True
        st.session_state[SS_USER_ID] = user["id"]
        st.session_state[SS_USERNAME] = user["username"]
        return True

    return render_login_page()


def render_login_page() -> bool:
    st.title("RV Prediction Lab")
    st.caption("Sign in to continue (shared-server mode).")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary", key="login_btn"):
            user = db.get_user_by_username(username.strip())
            if user and db.verify_password(password, user["password_hash"]):
                if user["username"] == db.LOCAL_USERNAME:
                    st.error("Reserved account.")
                else:
                    st.session_state[SS_AUTHENTICATED] = True
                    st.session_state[SS_USER_ID] = user["id"]
                    st.session_state[SS_USERNAME] = user["username"]
                    st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_register:
        expected = invite_code_expected()
        if not expected:
            st.warning("CLASS_INVITE_CODE is not set in .env — registration disabled.")
        else:
            r_user = st.text_input("New username", key="reg_user")
            r_pass = st.text_input("New password", type="password", key="reg_pass")
            invite = st.text_input("Invite code", type="password", key="reg_invite")
            if st.button("Register", key="reg_btn"):
                if invite.strip() != expected:
                    st.error("Invalid invite code.")
                elif len(r_user.strip()) < 3:
                    st.error("Username must be at least 3 characters.")
                elif len(r_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif db.get_user_by_username(r_user.strip()):
                    st.error("Username already taken.")
                else:
                    user = db.create_user(r_user.strip(), r_pass)
                    st.session_state[SS_AUTHENTICATED] = True
                    st.session_state[SS_USER_ID] = user["id"]
                    st.session_state[SS_USERNAME] = user["username"]
                    st.success("Account created.")
                    st.rerun()

    return bool(st.session_state.get(SS_AUTHENTICATED))


def render_workspace_sidebar() -> bool:
    """Sidebar workspace selector. Returns True when a workspace is active."""
    from project.src.streamlit_app.workspace import restore_workspace

    user_id = st.session_state.get(SS_USER_ID)
    if not user_id:
        return False

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**User:** {st.session_state.get(SS_USERNAME, '?')}")
    if st.sidebar.button("Log out", key="sidebar_logout_btn"):
        for key in (SS_AUTHENTICATED, SS_USER_ID, SS_USERNAME, "workspace_id"):
            st.session_state[key] = False if key == SS_AUTHENTICATED else None
        st.rerun()

    workspaces = db.list_workspaces(user_id)
    names = {w["id"]: w["name"] for w in workspaces}

    if not workspaces:
        st.sidebar.info("Create a workspace to begin.")
        new_name = st.sidebar.text_input("Workspace name", value="My Lab")
        if st.sidebar.button("Create workspace", type="primary"):
            ws = db.create_workspace(user_id, new_name.strip() or "My Lab")
            restore_workspace(ws["id"])
            st.rerun()
        return False

    ids = list(names.keys())
    current = st.session_state.get("workspace_id")
    try:
        index = ids.index(current) if current in ids else 0
    except ValueError:
        index = 0

    selected = st.sidebar.selectbox(
        "Workspace",
        options=ids,
        index=index,
        format_func=lambda i: names[i],
        key="ws_select",
    )
    if selected != current:
        restore_workspace(selected)
        st.rerun()

    if st.sidebar.button("New workspace"):
        ws = db.create_workspace(user_id, f"Lab {len(workspaces) + 1}")
        restore_workspace(ws["id"])
        st.rerun()

    rename_to = st.sidebar.text_input(
        "Rename",
        value=names.get(selected, ""),
        key="ws_rename",
    )
    if st.sidebar.button("Save name") and rename_to.strip():
        db.rename_workspace(selected, rename_to.strip())
        st.session_state["workspace_name"] = rename_to.strip()
        st.rerun()

    if st.sidebar.button("Save workspace now"):
        from project.src.streamlit_app.workspace import checkpoint_current_workspace

        checkpoint_current_workspace("manual save")

    st.sidebar.markdown("**Export / Import**")
    from project.src.streamlit_app.workspace import (
        export_workspace_zip_bytes,
        import_workspace_as_new,
        import_workspace_overwrite_current,
        parse_workspace_import,
    )

    if st.sidebar.button("Prepare export zip", key="ws_export_prepare"):
        try:
            st.session_state["_ws_export_zip"] = export_workspace_zip_bytes()
            st.sidebar.success("Export ready — click Download below.")
        except Exception as exc:
            st.session_state.pop("_ws_export_zip", None)
            st.sidebar.error(f"Export failed: {exc}")

    packed = st.session_state.get("_ws_export_zip")
    if packed:
        export_bytes, export_name = packed
        st.sidebar.download_button(
            "Download workspace zip",
            data=export_bytes,
            file_name=export_name,
            mime="application/zip",
            key="ws_export_btn",
        )

    uploaded = st.sidebar.file_uploader(
        "Import workspace zip",
        type=["zip"],
        key="ws_import_file",
    )
    import_mode = st.sidebar.radio(
        "On import",
        options=["new", "overwrite"],
        format_func=lambda m: (
            "Create new workspace" if m == "new" else "Overwrite current"
        ),
        key="ws_import_mode",
    )
    if uploaded is not None and st.sidebar.button("Apply import", key="ws_import_apply"):
        try:
            parsed = parse_workspace_import(uploaded.getvalue())
            zip_bytes = parsed.get("zip_bytes")
            if import_mode == "new":
                import_workspace_as_new(
                    user_id,
                    parsed["name"],
                    parsed["config"],
                    zip_bytes=zip_bytes,
                )
                st.toast(f"Imported as new workspace: {parsed['name']}")
            else:
                import_workspace_overwrite_current(
                    parsed["config"],
                    zip_bytes=zip_bytes,
                )
                st.toast("Current workspace overwritten from import")
            st.session_state.pop("_ws_export_zip", None)
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Import failed: {exc}")

    return st.session_state.get("workspace_id") is not None
