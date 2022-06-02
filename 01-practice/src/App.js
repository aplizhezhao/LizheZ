import React, {useState} from 'react';
import AddUser from './components/User/AddUser'
import UserList from './components/User/UserList'

function App() {
  const [usersList, setUsersList] = useState([]);
  
  const addUserHandler = (userName, userAge) => {
    setUsersList((prevUsersList) => {
      return ([...prevUsersList, {name: userName, age: userAge, id: Math.random()}])
    })
  }

  return (
    <div>
      <AddUser onAdd={addUserHandler}/>
      <UserList users={usersList}/>
    </div>
  );
}

export default App;
