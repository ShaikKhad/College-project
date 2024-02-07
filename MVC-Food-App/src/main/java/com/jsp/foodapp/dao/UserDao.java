package com.jsp.foodapp.dao;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.EntityTransaction;
import javax.persistence.NoResultException;
import javax.persistence.Query;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import com.jsp.foodapp.dto.User;

@Repository
public class UserDao {
	@Autowired
	EntityManagerFactory emf;

	public void saveUser(User u) {
		EntityManager entityManager = emf.createEntityManager();
		EntityTransaction entityTransaction = entityManager.getTransaction();

		entityTransaction.begin();
		entityManager.persist(u);
		entityTransaction.commit();
	}
	
	
	public User findUserByEmailAndPassword(String email, String password) {
		EntityManager em = emf.createEntityManager() ;
		Query query = em.createQuery("select u from User u where u.email=?1 and u.password=?2" ) ;
		query.setParameter(1, email) ;
		query.setParameter(2, password) ;
		
		try {
			 return (User)query.getSingleResult() ;
		}
		catch(NoResultException e) {
			return null ;
		}
		
	}
	
	public void updateUser(User u) {
		EntityManager entityManager = emf.createEntityManager();
		EntityTransaction entityTransaction = entityManager.getTransaction();

		entityTransaction.begin();
		entityManager.merge(u);
		entityTransaction.commit();
	}
}
