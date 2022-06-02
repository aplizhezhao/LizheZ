import ExpenseDate from "./ExpenseDate";
import Card from './Card'
import './ExpenseItem.css';

const ExpenseItem = (props) => {
  const deleteHandler = () => {
    if(window.confirm('Delete this expense?')) {
      props.onDelete(props.id)
    }
  }

  return (
    <li>
    <Card className='expense-item'>
        <ExpenseDate date={props.date} />
        <div className='expense-item__description'>
            <h2>{props.title}</h2>
            <div className='expense-item__price'>${props.amount}</div>
        </div>
        <button onClick={deleteHandler}>Delete</button>
    </Card>
    </li>
  );
}

export default ExpenseItem;
