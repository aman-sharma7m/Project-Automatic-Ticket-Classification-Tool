import streamlit as st


st.title('Departments')

tab_titles=['HR','IT','Transportation']

tabs=st.tabs(tab_titles)

with tabs[0]:
  st.header('HR Support tickets')

  for num,ticket in enumerate(list(set(st.session_state['HR_tickets']))):
    st.write(f'{num+1}: {ticket}')

with tabs[1]:
  st.header('IT Support tickets')

  for num,ticket in enumerate(list(set(st.session_state['IT_tickets']))):
    st.write(f'{num+1}: {ticket}')

with tabs[2]:
  st.header('Transportation Support tickets')

  for num,ticket in enumerate(list(set(st.session_state['TRANS_tickets']))):
    st.write(f'{num+1}: {ticket}')



