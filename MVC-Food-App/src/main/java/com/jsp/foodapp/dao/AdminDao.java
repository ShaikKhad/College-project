package com.jsp.foodapp.dao;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.EntityTransaction;
import javax.persistence.NoResultException;
import javax.persistence.Query;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import com.jsp.foodapp.dto.Admin;

@Repository
public class AdminDao {

	@Autowired
	EntityManagerFactory entityManagerFactory;

	public void saveAdmin(Admin admin) {
		EntityManager entityManager = entityManagerFactory.createEntityManager();
		EntityTransaction entityTransaction = entityManager.getTransaction();

		entityTransaction.begin();
		entityManager.persist(admin);
		entityTransaction.commit();
	}

	public Admin fetchAdminByEmailAndPassword(String email, String password) {
		EntityManager entityManager = entityManagerFactory.createEntityManager();

		Query query = entityManager.createQuery("select a from Admin a where a.email=?1 and a.password=?2");
         query.setParameter(1, email) ;
         query.setParameter(2, password) ;
         
         
         try { 
        	return (Admin)query.getSingleResult() ;
         }
         catch(NoResultException e) {
        	 return null ;      
         }
	}
}
