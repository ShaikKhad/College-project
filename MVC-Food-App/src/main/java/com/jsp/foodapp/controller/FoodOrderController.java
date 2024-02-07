package com.jsp.foodapp.controller;

import java.util.ArrayList;
import java.util.List;

import javax.servlet.http.HttpSession;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.servlet.ModelAndView;

import com.jsp.foodapp.dao.UserDao;
import com.jsp.foodapp.dto.FoodOrder;
import com.jsp.foodapp.dto.Item;
import com.jsp.foodapp.dto.User;

@Controller
public class FoodOrderController {
	
	@Autowired
	UserDao dao ;

	@RequestMapping("/addfoodorder")
	public ModelAndView addFoodOrder() {
		ModelAndView mav = new ModelAndView("addfoodorder");
		FoodOrder foodOrder = new FoodOrder();
		mav.addObject("foodorder", foodOrder);
		return mav;
	}

	@RequestMapping("/savefoodorder")
	public ModelAndView saveFoodOrder(@ModelAttribute("foodorder") FoodOrder order, HttpSession session) {
		ModelAndView mav = new ModelAndView("redirect:/viewallproduct");
		session.setAttribute("foodorder", order);
		return mav;
	}

	@RequestMapping("/savefoodorders")
	public ModelAndView saveFoodOrders(HttpSession session) {
	          User user = (User)      session.getAttribute("user") ;
	          FoodOrder foodorder = (FoodOrder) session.getAttribute("foodorder") ;
	          
	          List<FoodOrder> orders = user.getFoodorders() ;
	          if(orders!=null) {
	        	  orders.add(foodorder) ;
	        	  
	        	  double sum = 0 ;
	        	  List<Item> items = foodorder.getItems() ;
	        	  for(Item i : items) {
	        		  sum=sum+i.getCost() ;
	        	  }
	        	  
	        	  foodorder.setTotalcost(sum) ;
	        	  
	        	  user.setFoodorders(orders) ;
	        	  foodorder.setUser(user) ;
	          }
	          else {
	        	  List<FoodOrder> orders1 = new ArrayList<FoodOrder>() ;
	        	  orders1.add(foodorder) ;
	        	 
	        	  
	        	  double sum = 0 ;
	        	  List<Item> items = foodorder.getItems() ;
	        	  for(Item i : items) {
	        		  sum=sum+i.getCost() ;
	        	  }
	        	  
	        	  foodorder.setTotalcost(sum) ;
	        	  foodorder.setUser(user) ;
	        	  user.setFoodorders(orders1) ;
	        	  
	          }
	          
	          dao.updateUser(user) ;
	          
	          ModelAndView mav = new ModelAndView("displaybill") ;
	          return mav ;
	}
	
}
