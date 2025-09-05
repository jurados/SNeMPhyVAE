import datetime
import os
import requests
import json
import zipfile
import shutil
import time
import numpy as np
WISeREP                = "www.wiserep.org"

url_wis_spectra_search = "https://" + WISeREP + "/search/spectra"

# Specify the Personal api key here (*** MUST BE PROVIDED ***)
personal_api_key       = "YOUR_PERSONAL_API_KEY"

# for User-Agent:
WIS_USER_NAME          = "YOUR_USER_NAME"
WIS_USER_ID            = "YOUR_USER_ID"

# sn2017ixz, sn2017eaw, 2017iyd, 2017hbg, 2020buc

class SpectraWISREP:
	
	def __init__(self, sn_name: str):
		self.sn_name = sn_name
		
	def obtain_spectra(self):
		# Example of specific parameter options related to the download itself:
		# &num_page=250
		# &format=tsv/csv/json
		# &files_type=none/ascii/fits/all
		# &personal_api_key=...
		# &bot_api_key=...


		# Specify the required parameters here
		# Possible files type: &files_type=none | ascii | fits | all
		# Possible metadata list format: &format=csv | tsv | json
		# (In this example - all public type Ib/c's (and sub-types); metadata in CSV format; incl. ascii files.)

		# types are: 
		# {1: SN, 2: SN I, 3: SN Ia, 4: SN Ib, 5: SN Ic, 6: SN Ib/c,
		#  7: SN Ic-BL, 9: SN IIbn, 10: SN II, 11: SN IIP, 11: SN IIP,
		#  12: SN IIL, 13: SN IIn, 14: SN IIb, 15: SN I-faint, 16: SN I-rapid,
		#  18: SLSN-I, 19: SNLS-II, 111: SN II-pec, 112: SN IIn-pec, 104: SN Ia-91T-like, 103: SN Ia-91bg-like,
		#  }

		# spectypes:
		# {10: Objects, 50: Synthetic}
		#https://www.wiserep.org/search/spectra?&name=sn2017eaw&name_like=0&public=yes&inserted_period_value=0&inserted_period_units=months&type%5B%5D=99&type_family%5B%5D=1&instruments%5B%5D=157&spectypes%5B%5D=10&qualityid%5B%5D=3&groupid%5B%5D=null&spectra_count=&redshift_min=0&redshift_max=2&obsdate_start%5Bdate%5D=&obsdate_end%5Bdate%5D=&spec_phase_min=&spec_phase_max=&spec_phase_unit=days&phase_types%5B%5D=null&filters%5B%5D=null&methods%5B%5D=null&wl_min=&wl_max=&obj_ids=&spec_ids=&ids_or=0&reporters=&publish=&contrib=&last_modified_start%5Bdate%5D=&last_modified_end%5Bdate%5D=&last_modified_modifier=&creation_start%5Bdate%5D=&creation_end%5Bdate%5D=&creation_modifier=&show_aggregated_spectra=1&show_all_spectra=0&table_phase_name=40&num_page=250&display%5Bobj_rep_internal_name%5D=1&display%5Bobj_type_family_name%5D=0&display%5Bobj_type_name%5D=1&display%5Bredshift%5D=1&display%5Bphases%5D=1&display%5Bexptime%5D=1&display%5Bobserver%5D=1&display%5Breducers%5D=1&display%5Bsource_group_name%5D=1&display%5Basciifile%5D=1&display%5Bfitsfile%5D=1&display%5Bspectype_name%5D=1&display%5Bquality_name%5D=1&display%5Bextinction_corr_name%5D=0&display%5Bflux_calib_name%5D=0&display%5Bwl_medium_name%5D=0&display%5Bgroups%5D=0&display%5Bpublic%5D=1&display%5Bend_pop_period%5D=0&display%5Breporters%5D=0&display%5Bpublish%5D=1&display%5Bcontrib%5D=0&display%5Bremarks%5D=0&display%5Bcreatedby%5D=1&display%5Bcreationdate%5D=1&display%5Bmodifiedby%5D=0&display%5Blastmodified%5D=0
		# name=sn2017eaw&s
		types_sn = list(np.arange(1,20,1))
		query_params           = f"&public=yes&type[]="+str(types_sn)+"&spectypes[]=10"
		#query_params           = "&public=yes&type[]=[10,11]&spectypes[]=10"
		#query_params           = f"&public=yes&name={self.sn_name}&spectypes%5B%5D=10"
		download_params        = "&num_page=250&format=csv&files_type=ascii"

		parameters             = "?" + query_params+download_params + "&personal_api_key=" + personal_api_key
		print(f'Parameters: {parameters}')

		# url of wiserep spectra search (with parameters)
		URL                    = url_wis_spectra_search + parameters

		return download_params, URL
		
		
		#------------------------------------------------------------------------


	#------------------------------------------------------------------------
	def is_string_json(self, string):
		try:
			json_object = json.loads(string)
		except Exception:
			return False
		return json_object

	def response_status(self,response):
		# external http errors
		ext_http_errors       = [403, 500, 503]
		err_msg               = ["Forbidden", "Internal Server Error: Something is broken", "Service Unavailable"]

		json_string = self.is_string_json(response.text)
		if json_string != False:
			status = "[ " + str(json_string['id_code']) + " - '" + json_string['id_message'] + "' ]"
		else:
			status_code = response.status_code
		if status_code == 200:
			status_msg = 'OK'
		elif status_code in ext_http_errors:
			status_msg = err_msg[ext_http_errors.index(status_code)]
		else:
			status_msg = 'Undocumented error'
		status = "[ " + str(status_code) + " - '" + status_msg + "' ]"
		return status

	def print_response(self, response, page_num):
		status = self.response_status(response)
		stats = 'Page number ' + str(page_num) + ' | return code: ' + status        
		print(stats)
		#------------------------------------------------------------------------

	def download_spectra(self):
		download_params, URL = self.obtain_spectra()
		#------------------------------------------------------------------------
		# current date and time
		current_datetime = datetime.datetime.now()
		current_date_time = current_datetime.strftime("%Y%m%d_%H%M%S")

		# current working directory
		cwd = os.getcwd()
		# current download folder
		current_download_folder = os.path.join(cwd+"/data/spectra", "wiserep_data_" + self.sn_name)
		os.mkdir(current_download_folder)

		# marker and headers
		wis_marker = 'wis_marker{"wis_id": "' + str(WIS_USER_ID) + '", "type": "user", "name": "' + WIS_USER_NAME + '"}'
		headers = {'User-Agent': wis_marker}

		# check file extension
		if "format=tsv" in download_params:
			extension = ".tsv"
		elif "format=csv" in download_params:
			extension = ".csv"
		elif "format=json" in download_params:
			extension = ".json"
		else:
			extension = ".txt"

		# meta data list and file
		META_DATA_LIST = []
		META_DATA_FILE = os.path.join(cwd+"/data/spectra", "spectra_" + self.sn_name + extension)

		# page number
		page_num = 0

		# go trough every page
		
		while True:
			# url for download
			url = URL + "&page=" + str(page_num)
			# send requests
			response = requests.post(url, headers = headers, stream = True)
			# chek if response status code is not 200
			if response.status_code != 200:
			# if there are no more pages for download, don't print response, 
			# only print if response is something else
				if response.status_code != 404:
					self.print_response(response, page_num + 1)
				break 
			# print response
			self.print_response(response, page_num + 1)
			# download data
			file_name = 'wiserep_spectra.zip'
			file_path = os.path.join(current_download_folder, file_name)
			with open(file_path, 'wb') as f:
				for data in response:
					f.write(data)
			# unzip data
			zip_ref = zipfile.ZipFile(file_path, 'r')
			zip_ref.extractall(current_download_folder)
			zip_ref.close()
			# remove .zip file
			os.remove(file_path)            
			# take meta data file
			downloaded_files = os.listdir(current_download_folder)
			meta_data_file = os.path.join(current_download_folder, [e for e in downloaded_files if 'wiserep_spectra' in e][0])          
			# read meta data file
			f = open(meta_data_file,'r')
			meta_data_list = f.read().splitlines()
			f.close()
			# write this meta data list to the final meta data list
			if page_num == 0:
				META_DATA_LIST = META_DATA_LIST + meta_data_list
			else:
				META_DATA_LIST = META_DATA_LIST + meta_data_list[1:]         
			# increase page number 
			page_num = page_num + 1                 
			# remove meta data file
			os.remove(meta_data_file)

		# write meta data list to file         
		print('Salimos del While')
		if META_DATA_LIST != []:
			f = open(META_DATA_FILE, 'w')
			for i in range(len(META_DATA_LIST)):
				if i == len(META_DATA_LIST) - 1:
					f.write(META_DATA_LIST[i])
				else:
					f.write(META_DATA_LIST[i] + '\n')
			f.close()
			print ("Wiserep data was successfully downloaded.")
			print ("Folder /wiserep_data_" + self.sn_name + "/ containing the data was created.")
			print ("File spectra_" + self.sn_name + extension + " was created.")
		else:
			print ("There is no WISeREP data for the given parameters.")
			shutil.rmtree(current_download_folder)
		#------------------------------------------------------------------------